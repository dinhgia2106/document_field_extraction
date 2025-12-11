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
from collections import Counter
from PIL import Image
from pdf2image import convert_from_path
import google.generativeai as genai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
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
EXTRACTION_MODEL = "gemini-2.5-flash"
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
    
    def __init__(self, schema: Any, model_name: str = EXTRACTION_MODEL, is_dynamic: bool = False):
        """
        Args:
            schema: Either a Pydantic BaseModel class (static) or a Dict (dynamic schema definition)
            model_name: Gemini model to use
            is_dynamic: Helper flag to switch validation logic
        """
        self.model = genai.GenerativeModel(model_name)
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
        
        # Initialize OpenAI client 
        self.openai_client = None
        if OpenAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
            else:
                print("[SchemaExtractor] Warning: OPENAI_API_KEY not found. ChatGPT calls will be skipped.")
        else:
             print("[SchemaExtractor] Warning: 'openai' module not installed. ChatGPT calls will be skipped.")

    def _build_extraction_prompt(self, markdown_text: str, error_feedback: Optional[str] = None) -> str:
        """Build the extraction prompt with optional error feedback for self-correction."""
        
        base_prompt = f"""
You are a data extraction specialist. Extract structured information from the {self.doc_type_hint} below.

You must output a JSON object with two top-level keys:
1. "data": The extracted data matching the TARGET SCHEMA.
2. "field_evaluations": A dictionary where keys are field names. For EACH extracted field, providing:
   - "confidence": A score from 0.0 to 1.0 indicating your certainty.
   - "relevance_check": A brief check on whether this extracted value is truly relevant to the field's definition in the context of this document.

TARGET SCHEMA for "data":
```json
{self.schema_json}
```

EXTRACTION RULES:
- STRICT JSON: Return ONLY a valid JSON object with "data" and "field_evaluations".
- MISSING DATA: Use null for missing values in "data".
- NUMBERS: Extract numbers purely (e.g., "20,025" -> 20025).
- IDS: Look for "Access Code" and "Notice Number" usually found in the top right header area.
- CRITICAL: Review the "relevance_check" carefully. If a field seems present but doesn't semantically match the schema definition, lower the confidence.

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
        """Call Gemini and parse JSON response."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"[Extractor] Gemini call failed: {e}")
            return {}

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call ChatGPT (OpenAI) and parse JSON response."""
        if not self.openai_client:
            return {}
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"[Extractor] ChatGPT call failed: {e}")
            return {}

    def extract(self, markdown_text: str, with_confidence: bool = False) -> Dict[str, Any]:
        """
        Extracts structured data using a single Gemini pass with self-evaluation.
        """
        print("[Extractor] Starting Extraction (Gemini + Self-Correction)...")
        
        prompt = self._build_extraction_prompt(markdown_text)
        
        # Single Gemini Call
        result = self._call_llm(prompt)
        
        if not result:
            print("[Extractor] Extraction failed (No response).")
            return {}
        
        # Normalize result structure
        if "data" in result and "field_evaluations" in result:
             final_data = result["data"]
             evals = result["field_evaluations"]
             
             # Attach metadata
             final_data["_confidence"] = {k: v.get("confidence", 0.0) for k, v in evals.items()}
             final_data["_relevance"] = {k: v.get("relevance_check", "") for k, v in evals.items()}
             
             return final_data
        elif "data" in result:
             # Partial success
             return result["data"]
        else:
             # Fallback if LLM forgot structure and just returned data
             # Check if it looks like data (heuristic)
             print("[Extractor] Warning: Unexpected response structure, assuming direct data.")
             return result

    def _format_validation_error(self, error: ValidationError) -> str:
        """Format Pydantic validation errors for LLM feedback."""
        errors = []
        for e in error.errors():
            field = " -> ".join(str(x) for x in e["loc"])
            errors.append(f"- Field '{field}': {e['msg']} (got: {e.get('input', 'N/A')})")
        return "\n".join(errors)
    
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
