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
import typing
from typing import List, Dict, Any, Optional, Type
from PIL import Image
from pdf2image import convert_from_path
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
import re

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
        print(f"[Ingestion] Loading PDF: {pdf_path}")
        print(f"[Ingestion] Using DPI: {dpi}")
        
        try:
            images = convert_from_path(pdf_path, dpi=dpi)
            print(f"[Ingestion] Converted {len(images)} pages to high-res images")
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
    confidence: float = 0.8


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
{"classification": "A", "reason": "brief explanation", "confidence": 0.9}
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
                "reason": result.get("reason", "Unknown"),
                "confidence": result.get("confidence", 0.5)
            }
        except Exception as e:
            print(f"[Router] Warning: Error classifying page {page_num}: {e}")
            # FAIL-SAFE: Keep page when uncertain (human expert approach)
            return {
                "page": page_num,
                "classification": "B",  # Keep by default
                "reason": f"Classification failed, keeping as safety measure",
                "confidence": 0.0
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
1. Use standard Markdown tables (| and -) for ANY tabular data - this is critical
2. Use # ## ### for headers based on visual hierarchy
3. Preserve all numbers, dates, and amounts EXACTLY as shown
4. Keep the spatial relationship between labels and values
5. For key-value pairs, use bold for labels: **Label:** Value
6. Ignore repetitive headers/footers (page numbers, document IDs) unless they contain unique info
7. Do NOT summarize - transcribe content accurately and completely

OUTPUT: Clean Markdown text only, no explanations.
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
You are a Senior Data Architect. Analyze these document pages (first few pages) to identify the ideal data structure.

YOUR TASK:
1. Identify the Document Type (e.g., Invoice, Resume, Contract, Tax Form).
2. Determine the 10-15 MOST CRITICAL fields that represent the core value of this document.
   - Look across ALL pages to find relevant data.
   - For Tax/Financial documents, MUST include: Total Income, Net Income, Taxable Income, Total Payable, Deductions, Credits.
   - Focus on: ID numbers, Dates, Total Amounts, People/Entity Names.
   - Do NOT include generic fields like "page_number" or layout details.
3. Define a JSON Schema for these fields.

RETURN ONLY A VALID JSON OBJECT WITH THIS STRUCTURE:
{
    "document_type": "Name of document type",
    "description": "Brief description of the document's purpose",
    "fields": [
        {
            "name": "field_name_snake_case",
            "type": "string|number|date|boolean",
            "description": "What this field represents"
        }
    ]
}
"""

    def discover(self, images: List[Image.Image]) -> Dict[str, Any]:
        """
        Analyzes the first few pages to generate a dynamic schema.
        """
        # Limit to first 3 pages to get context without overloading
        sample_images = images[:3]
        print(f"[Discovery] Analyzing {len(sample_images)} pages for document structure...")
        
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
                    {"name": "total_amount", "type": "number", "description": "Total value or amount if applicable"},
                    {"name": "date", "type": "string", "description": "Primary date of the document"},
                    {"name": "entities", "type": "array", "description": "Names of people or organizations"}
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
    
    def _build_extraction_prompt(self, markdown_text: str, error_feedback: Optional[str] = None) -> str:
        """Build the extraction prompt with optional error feedback for self-correction."""
        
        base_prompt = f"""
You are a data extraction specialist. Extract structured information from the {self.doc_type_hint} below.

TARGET SCHEMA (you MUST return data matching this structure):
```json
{self.schema_json}
```

EXTRACTION RULES:
1. Return ONLY a valid JSON object matching the schema above.
2. Use null for missing or unclear values.
3. Convert currency strings to numbers (e.g., "$1,234.56 CR" -> -1234.56).
4. Dates should be in ISO format (YYYY-MM-DD) when possible.
5. Be precise - extract exactly what the document says.

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
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)

    def extract(self, markdown_text: str) -> Dict[str, Any]:
        """
        Extracts structured data from markdown with self-correction loop.
        
        Args:
            markdown_text: Normalized Markdown content
            
        Returns:
            Validated dictionary matching the schema
        """
        print(f"[Extractor] Starting extraction with {self.MAX_RETRIES} max attempts...")
        
        error_feedback = None
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"[Extractor] Attempt {attempt}/{self.MAX_RETRIES}...")
            
            try:
                # Build prompt (with error feedback if retry)
                prompt = self._build_extraction_prompt(markdown_text, error_feedback)
                
                # Call LLM
                raw_result = self._call_llm(prompt)
                print(f"[Extractor] LLM returned: {json.dumps(raw_result, indent=2)[:200]}...")
                
                # Validate
                if self.is_dynamic:
                    # For dynamic mode, we just check if it's valid JSON (already done by json.loads)
                    # We could add more complex checks here if needed
                    print(f"[Extractor] Validation passed (Dynamic Mode)")
                    return raw_result
                else:
                    # Startic Pydantic Validation
                    validated = self.schema_obj.model_validate(raw_result)
                    print(f"[Extractor] Validation passed on attempt {attempt}")
                    return validated.model_dump()
                
                
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
