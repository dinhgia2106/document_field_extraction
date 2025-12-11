"""
Document Field Extraction Pipeline - Modules
=============================================
4-Module Architecture:
1. Ingestion & Vision Encoder - Convert PDF to high-res images
2. Semantic Router - Classify and filter pages (A/B/C)
3. Structural Normalizer - Convert images to Markdown
4. Schema Extractor - Extract structured data with self-correction
"""

import os
import json
import math
import typing
from typing import List, Dict, Any, Optional, Type
from PIL import Image
from pdf2image import convert_from_path
from openai import OpenAI
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
import re
from difflib import SequenceMatcher

# Configure Gemini
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# =============================================================================
# MODEL CONFIGURATION - Tiered approach
# =============================================================================
# Fast/cheap model for routing (classification tasks)
ROUTER_MODEL = "gemini-2.0-flash"
# Powerful model for extraction (complex reasoning)
EXTRACTION_MODEL = "gemini-2.0-flash" 
OPENAI_MODEL = "gpt-4o-mini"
# High DPI for better OCR accuracy
IMAGE_DPI = 300


# =============================================================================
# MODULE 1: INGESTION & VISION ENCODER
# =============================================================================
class Ingestion:
    """
    Module 1: Ingestion & Vision Encoder
    
    Theory:
    - Spatial layout contains 50% of document meaning
    - Convert PDF to High-Res Images instead of plain text
    - Preserve spatial relationships between elements
    """
    
    @staticmethod
    def process(pdf_path: str, dpi: int = IMAGE_DPI) -> List[Image.Image]:
        """
        Converts a PDF into a list of high-resolution PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion (default: 300 for high quality)
            
        Returns:
            List of PIL Images, one per page
        """
        # print(f"[Ingestion] Loading PDF: {pdf_path}")
        # print(f"[Ingestion] Using DPI: {dpi}")
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            # print(f"[Ingestion] Converted {len(images)} pages to high-res images")
            return images
        except Exception as e:
            print(f"[Ingestion] Error converting PDF: {e}")
            raise e


# =============================================================================
# MODULE 2: SEMANTIC ROUTER
# =============================================================================
class PageClassification(BaseModel):
    """Schema for page classification result"""
    classification: str  # A, B, or C
    reason: str


class SemanticRouter:
    """
    Module 2: Semantic Router
    
    Problems solved:
    - Avoid Context Overflow for long documents
    - Reduce signal noise (improve Signal-to-Noise Ratio)
    
    Classification:
    - Class A (Target): Calculation tables, Summary, Total Payable -> KEEP
    - Class B (Info): Address info, Account numbers -> KEEP  
    - Class C (Noise): Definitions, Instructions, Blank pages -> DROP
    """
    
    def __init__(self, model_name: str = ROUTER_MODEL):
        self.model = genai.GenerativeModel(model_name)
        self.classification_prompt = """
You are a document page classifier. Analyze this document page image and classify it.

CLASSIFICATION RULES:
- Class A (HIGH PRIORITY - KEEP): Contains numeric tables, financial summaries, tax calculations, 
  total amounts, assessment results, payment information, or key numerical data.
- Class B (MEDIUM PRIORITY - KEEP): Contains personal information, taxpayer name, address, 
  account numbers, reference numbers, dates, or identification details.
- Class C (LOW PRIORITY - DROP): Contains only instructions, definitions, legal disclaimers, 
  terms and conditions, blank pages, or purely decorative content.

IMPORTANT: When in doubt between keeping or dropping, classify as A or B (prefer to keep).

Return ONLY a valid JSON object in this exact format:
{"classification": "A", "reason": "brief explanation"}
"""

    def _classify_page(self, image: Image.Image, page_num: int) -> Dict[str, Any]:
        """Classify a single page with error handling."""
        try:
            response = self.model.generate_content([self.classification_prompt, image])
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            return {
                "page": page_num,
                "classification": result.get("classification", "B"),  # Default to keep
                "reason": result.get("reason", "Unknown")
            }
        except Exception as e:
            print(f"[Router] Warning: Error classifying page {page_num}: {e}")
            # FAIL-SAFE: Keep page when uncertain (human expert approach)
            return {
                "page": page_num,
                "classification": "B",  # Keep by default
                "reason": f"Classification failed, keeping as safety measure"
            }

    def route(self, images: List[Image.Image]) -> tuple[List[int], List[Dict]]:
        """
        Analyzes pages and returns indices of pages to keep.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tuple of (kept_indices, classification_results)
        """
        print(f"[Router] Analyzing {len(images)} pages...")
        
        kept_indices = []
        results = []
        
        for i, img in enumerate(images):
            result = self._classify_page(img, i + 1)
            results.append(result)
            
            cls = result["classification"]
            status = "KEEP" if cls in ["A", "B"] else "DROP"
            print(f"[Router] Page {i+1}: Class {cls} ({status}) - {result['reason']}")
            
            if cls in ["A", "B"]:
                kept_indices.append(i)
        
        dropped = len(images) - len(kept_indices)
        print(f"[Router] Selected {len(kept_indices)} pages, dropped {dropped} pages")
        
        return kept_indices, results


# =============================================================================
# MODULE 3: STRUCTURAL NORMALIZER
# =============================================================================
class StructuralNormalizer:
    """
    Module 3: Structural Normalization (Convert to Markdown)
    
    Theory:
    - LLMs are trained on large amounts of Markdown documents
    - Markdown represents tables efficiently with fewer tokens
    - Preserves row/column structure for LLM context understanding
    """
    
    def __init__(self, model_name: str = EXTRACTION_MODEL):
        self.model = genai.GenerativeModel(model_name)
        self.normalize_prompt = """
You are a document transcription expert. Convert this document page to clean Markdown.

TRANSCRIPTION RULES:
1. **CAPTURE HEADERS:** ALWAYS transcribe the Name and Address block found at the top left/right of the page. This is MANDATORY.
2. **TABLES:** Use standard Markdown tables (| and -) for ANY tabular data.
3. **EXACTNESS:** Preserve all numbers, codes (e.g., Access Code), and amounts EXACTLY as shown.
4. **LABELS:** For key-value pairs, use bold for labels: **Label:** Value.
5. **LINE NUMBERS:** If a line has a number (e.g., 15000, 43500), ensure it is written next to the description.

OUTPUT: Clean Markdown text only.
"""

    def normalize(self, images: List[Image.Image]) -> str:
        """
        Converts selected images to a single Markdown document.
        
        Args:
            images: List of PIL Images to normalize
            
        Returns:
            Combined Markdown string
        """
        print(f"[Normalizer] Converting {len(images)} pages to Markdown...")
        
        markdown_pages = []
        
        for i, img in enumerate(images):
            try:
                response = self.model.generate_content([self.normalize_prompt, img])
                page_md = response.text.strip()
                
                # Add page separator
                markdown_pages.append(f"<!-- Page {i+1} -->\n{page_md}")
                print(f"[Normalizer] Page {i+1} converted ({len(page_md)} chars)")
                
            except Exception as e:
                print(f"[Normalizer] Error on page {i+1}: {e}")
                markdown_pages.append(f"<!-- Page {i+1} - Error: {e} -->")
        
        full_markdown = "\n\n---\n\n".join(markdown_pages)
        print(f"[Normalizer] Total Markdown: {len(full_markdown)} characters")
        
        return full_markdown


# =============================================================================
# MODULE 4: SCHEMA EXTRACTOR & DISCOVERY
# =============================================================================
class SchemaDiscovery:
    """
    Module 4a: Dynamic Schema Discovery
    
    Theory:
    - Instead of hardcoding schemas, let LLM "read" the document first
    - Identify document type and critical fields automatically
    - Generates a 'contract' (JSON Schema) for the extraction phase
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.discovery_prompt = """
You are a Senior Data Architect. Analyze these representative document pages to identify the comprehensive data structure.

YOUR TASK:
1. Identify the Document Type (e.g., Invoice, Resume, Contract, Tax Form).
2. Identify ALL fields that are relevant to the main content and purpose of the document.
   - Do NOT limit the number of fields. If a piece of information is relevant, include it.
   - Look across ALL provided pages.
   - For Tax/Financial documents: Include comprehensive details like specific line items, breakdown of taxes, credits, and future planning figures if present.
   - Focus on: ID numbers, Dates, Amounts, Names, Addresses, and Statuses.
   - Exclude: Page numbers, purely decorative text, or generic boilerplate instructions.
3. Define a JSON Schema for these fields.

RETURN ONLY A VALID JSON OBJECT WITH THIS STRUCTURE:
{
    "document_type": "Name of document type",
    "description": "Brief description of the document's purpose",
    "fields": [
        {
            "name": "field_name_snake_case",
            "type": "string|number|date|boolean|array",
            "description": "What this field represents"
        }
    ]
}
"""

    def discover(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Analyzes representative pages to generate a dynamic schema.
        Strategy: Uniform spread sampling for large documents to catch all section types.
        """
        total_pages = len(images)
        SAMPLE_LIMIT = 20  # Look at up to 20 pages to ensure coverage for 100+ page docs
        
        # Smart Sampling Strategy
        if total_pages <= SAMPLE_LIMIT:
            # Small/Medium document: Use all pages
            sample_images = images
            indices = set(range(total_pages))
        else:
            # Large document: Intelligent Distributed Sampling
            # Always keep first 3 pages (Intro/Summary/Table of Conteonts)
            indices = {0, 1, 2}
            # Always keep last 2 pages (Totals/Signatures)
            indices.add(total_pages - 1)
            indices.add(total_pages - 2)
            
            # Spread the remaining budget across the middle
            remaining_slots = SAMPLE_LIMIT - len(indices)
            if remaining_slots > 0:
                # Calculate stride to skip through the middle section
                start_mid = 3
                end_mid = total_pages - 2
                step = max(1, (end_mid - start_mid) // remaining_slots)
                
                for i in range(start_mid, end_mid, step):
                    indices.add(i)
            
            # Sort indices and extract images
            sample_images = [images[i] for i in sorted(list(indices))][:SAMPLE_LIMIT]

        print(f"[Discovery] Analyzing {len(sample_images)} representative pages (Indices: {[i+1 for i in sorted(list(indices))][:SAMPLE_LIMIT]})...")
        
        try:
            # Send prompt + images
            content = [self.discovery_prompt] + sample_images
            response = self.model.generate_content(content)
            text = response.text.strip()
            
            # Clean up code blocks
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            
            schema_def = json.loads(text.strip())
            print(f"[Discovery] Detected type: {schema_def.get('document_type', 'Unknown')}")
            fields = [f['name'] for f in schema_def.get('fields', [])]
            print(f"[Discovery] Proposed fields ({len(fields)}): {', '.join(fields)}")
            return schema_def
            
        except Exception as e:
            print(f"[Discovery] Error discovering schema: {e}")
            # Fallback schema
            return {
                "document_type": "General Document",
                "fields": [
                    {"name": "summary", "type": "string", "description": "Summary of the document content"},
                    {"name": "full_text", "type": "string", "description": "Full text content"},
                    {"name": "key_dates", "type": "array", "description": "List of important dates found"},
                    {"name": "total_amounts", "type": "array", "description": "List of financial amounts found"}
                ]
            }


class SchemaExtractor:
    """
    Module 4b: Schema-Driven Extraction
    
    Theory:
    - Don't ask LLM to "summarize", ask it to "fill out this Form"
    - Use Pydantic Schema for Structured Output
    - Self-Correction: Validation -> Error Feedback -> Retry
    """
    
    MAX_RETRIES = 3
    
    def __init__(self, schema: Any, model_name: str = OPENAI_MODEL, is_dynamic: bool = False):
        """
        Args:
            schema: Either a Pydantic BaseModel class (static) or a Dict (dynamic schema definition)
            model_name: OpenAI model to use
            is_dynamic: Helper flag to switch validation logic
        """
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.is_dynamic = is_dynamic
        self.schema_obj = schema
        
        if self.is_dynamic:
            # Convert the dynamic schema definition into a clean JSON structure target
            # We want to feed the LLM a sample structure it should mimic
            fields = schema.get('fields', [])
            structure = {}
            for f in fields:
                structure[f['name']] = f.get('description', '')
            
            self.schema_json = json.dumps(structure, indent=2)
            self.doc_type_hint = schema.get('document_type', 'Document')
        else:
            # Static Pydantic Model
            self.schema_json = json.dumps(schema.model_json_schema(), indent=2)
            self.doc_type_hint = "Document"

    def _build_extraction_prompt(self, markdown_text: str, error_feedback: Optional[str] = None) -> str:
        """Build the extraction prompt with optional error feedback for self-correction."""
        
        base_prompt = f"""
You are a data extraction specialist. Extract structured information from the {self.doc_type_hint} below.

TARGET SCHEMA (you MUST return data matching this structure):
```json
{self.schema_json}
```
EXTRACTION RULES:
STRICT JSON: Return ONLY a valid JSON object matching the schema above.
MISSING DATA: Use null for missing values.
NUMBERS: Extract numbers purely (e.g., "20,025" -> 20025).
CREDIT/DEBIT: Do NOT convert "CR" amounts to negative numbers automatically. Put the absolute amount in 'amount' and "CR" in 'balance_type'.
IDS: Look for "Access Code" and "Notice Number" usually found in the top right header area.
LINE ITEMS: Match descriptions to Line IDs (e.g., "Total income" -> 15000) even if the text slightly varies.

DOCUMENT TEXT:
---
{markdown_text}
---
"""
        
        if error_feedback:
            base_prompt += f"""

WARNING - PREVIOUS ATTEMPT FAILED VALIDATION:
{error_feedback}

Please fix these errors and try again. Ensure your response is valid JSON matching the schema.
"""
        return base_prompt

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse JSON response."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        content = response.choices[0].message.content
        return json.loads(content)
    
    def _call_llm_with_logprobs(self, prompt: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Call LLM and return both parsed JSON result and logprobs data.
        
        Returns:
            Tuple of (parsed_data, logprobs_list)
            logprobs_list contains [{"token": str, "prob": float, "logprob": float}, ...]
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            logprobs=True,
            top_logprobs=1 # We only needs the top one
        )
        
        content = response.choices[0].message.content
        parsed_json = json.loads(content)
        
        # Extract logprobs data from response
        logprobs_list = []
        logprobs_content = response.choices[0].logprobs.content
        
        if logprobs_content:
            for item in logprobs_content:
                prob = math.exp(item.logprob)
                logprobs_list.append({
                    "token": item.token,
                    "prob": prob,
                    "logprob": item.logprob
                })
        
        return parsed_json, logprobs_list
    
    def _calculate_field_confidence(self, extracted_data: Dict[str, Any], 
                                     logprobs_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate confidence score for each field based on logprobs.
        
        Strategy:
        1. Reconstruct the JSON string from tokens
        2. For each field key, find the token range where its value is defined
        3. Average the probabilities of those tokens
        
        Args:
            extracted_data: The extracted data dictionary
            logprobs_list: List of token logprobs from LLM response
            
        Returns:
            Dict mapping field names to confidence scores (0.0 - 1.0)
        """
        if not logprobs_list:
            return {}
        
        confidence_map = {}
        
        # Reconstruct the full response text from tokens
        full_text = "".join([lp["token"] for lp in logprobs_list])
        
        def get_field_confidence(field_name: str, field_value: Any) -> float:
            """
            Find the tokens corresponding to a field value and compute average confidence.
            """
            if field_value is None:
                return 0.0  # Null values have no confidence
            
            # Convert value to string representation for matching
            if isinstance(field_value, bool):
                value_str = "true" if field_value else "false"
            elif isinstance(field_value, (int, float)):
                value_str = str(field_value)
            elif isinstance(field_value, str):
                value_str = field_value
            elif isinstance(field_value, dict):
                # For nested objects, we'll calculate confidence recursively
                nested_confidences = []
                for k, v in field_value.items():
                    nested_conf = get_field_confidence(k, v)
                    if nested_conf > 0:
                        nested_confidences.append(nested_conf)
                return sum(nested_confidences) / len(nested_confidences) if nested_confidences else 0.0
            elif isinstance(field_value, list):
                # For arrays, average the confidence of each element
                if not field_value:
                    return 0.0
                element_confidences = []
                for item in field_value:
                    item_conf = get_field_confidence("", item)
                    if item_conf > 0:
                        element_confidences.append(item_conf)
                return sum(element_confidences) / len(element_confidences) if element_confidences else 0.0
            else:
                value_str = str(field_value)
            
            # Find the value in the full text and get corresponding token indices
            value_start = full_text.find(value_str)
            if value_start == -1:
                # Try without quotes for string values
                value_start = full_text.find(f'"{value_str}"')
                if value_start != -1:
                    value_start += 1  # Skip opening quote
            
            if value_start == -1:
                return 0.85  # Default confidence if we can't locate the value
            
            value_end = value_start + len(value_str)
            
            # Map character positions to token indices
            char_pos = 0
            token_probs = []
            
            for lp in logprobs_list:
                token_len = len(lp["token"])
                token_end = char_pos + token_len
                
                # Check if this token overlaps with the value range
                if char_pos < value_end and token_end > value_start:
                    token_probs.append(lp["prob"])
                
                char_pos = token_end
                
                if char_pos >= value_end:
                    break
            
            if token_probs:
                return sum(token_probs) / len(token_probs)
            
            return 0.85  # Default if no matching tokens found
        
        # Calculate confidence for each field
        def process_fields(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
            result = {}
            for key, value in data.items():
                field_path = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Recursively process nested objects
                    nested_result = process_fields(value, field_path)
                    result.update(nested_result)
                    # Also store the aggregate confidence for the parent
                    if nested_result:
                        result[field_path] = sum(nested_result.values()) / len(nested_result)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # List of objects
                    list_confidences = []
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_result = process_fields(item, f"{field_path}[{i}]")
                            result.update(item_result)
                            if item_result:
                                list_confidences.append(sum(item_result.values()) / len(item_result))
                    if list_confidences:
                        result[field_path] = sum(list_confidences) / len(list_confidences)
                else:
                    result[field_path] = get_field_confidence(key, value)
            
            return result
        
        confidence_map = process_fields(extracted_data)
        return confidence_map

    def extract(self, markdown_text: str, with_confidence: bool = False) -> Dict[str, Any]:
        """
        Extracts structured data from markdown with self-correction loop.
        
        Args:
            markdown_text: Normalized Markdown content
            with_confidence: If True, also calculate and return confidence scores
            
        Returns:
            Dict: Validated Data Dict. If with_confidence=True, includes '_confidence' key
                  with per-field confidence scores.
        """
        print(f"[Extractor] Starting extraction with {self.MAX_RETRIES} max attempts...")
        if with_confidence:
            print(f"[Extractor] Confidence calculation enabled (using logprobs)")
        
        error_feedback = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"[Extractor] Attempt {attempt}/{self.MAX_RETRIES}...")
            
            try:
                # Build prompt (with error feedback if retry)
                prompt = self._build_extraction_prompt(markdown_text, error_feedback)
                
                # Call LLM - with logprobs if confidence is requested
                if with_confidence:
                    raw_result, logprobs_list = self._call_llm_with_logprobs(prompt)
                    print(f"[Extractor] Received {len(logprobs_list)} tokens with logprobs")
                else:
                    raw_result = self._call_llm(prompt)
                    logprobs_list = None
                
                # print(f"[Extractor] LLM returned: {json.dumps(raw_result, indent=2)[:200]}...")
                
                # Validate
                if self.is_dynamic:
                    # For dynamic mode, we just check if it's valid JSON (already done by json.loads)
                    print(f"[Extractor] Validation passed (Dynamic Mode)")
                    
                    # Calculate confidence if requested
                    if with_confidence and logprobs_list:
                        confidence_map = self._calculate_field_confidence(raw_result, logprobs_list)
                        raw_result['_confidence'] = confidence_map
                        print(f"[Extractor] Calculated confidence for {len(confidence_map)} fields")
                    
                    return raw_result
                else:
                    # Static Pydantic Validation
                    validated = self.schema_obj.model_validate(raw_result)
                    print(f"[Extractor] Validation passed on attempt {attempt}")
                    data = validated.model_dump()
                    
                    # Calculate confidence if requested
                    if with_confidence and logprobs_list:
                        confidence_map = self._calculate_field_confidence(data, logprobs_list)
                        data['_confidence'] = confidence_map
                        print(f"[Extractor] Calculated confidence for {len(confidence_map)} fields")
                    
                    return data
                
                
            except ValidationError as ve:
                error_feedback = self._format_validation_error(ve)
                print(f"[Extractor] Validation failed: {error_feedback}")
                
            except json.JSONDecodeError as je:
                error_feedback = f"Invalid JSON response: {str(je)}"
                print(f"[Extractor] JSON parse error: {error_feedback}")
                
            except Exception as e:
                error_feedback = f"Unexpected error: {str(e)}"
                print(f"[Extractor] Error: {error_feedback}")
        
        # All retries failed
        print(f"[Extractor] Extraction failed after {self.MAX_RETRIES} attempts")
        return {}
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """Format Pydantic validation errors for LLM feedback."""
        errors = []
        for e in error.errors():
            field = " -> ".join(str(x) for x in e["loc"])
            errors.append(f"- Field '{field}': {e['msg']} (got: {e.get('input', 'N/A')})")
        return "\n".join(errors)


# =============================================================================
# UTILITY: PIPELINE RESULT
# =============================================================================
class PipelineResult(BaseModel):
    """Container for complete pipeline execution results."""
    success: bool
    total_pages: int
    kept_pages: int
    dropped_pages: int
    routing_details: List[Dict[str, Any]]
    markdown_preview: str
    extracted_data: Dict[str, Any]
    errors: List[str] = []