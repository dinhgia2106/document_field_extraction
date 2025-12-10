"""
Document Field Extraction Pipeline
====================================
Process documents like an expert: Scan -> Select important pages -> Read carefully -> Take notes

Usage:
    python pipeline.py --pdf "path/to/document.pdf"
    python pipeline.py --pdf "NOA 2023.pdf" --verbose
    python pipeline.py --pdf "NOA 2023.pdf" --output result.json
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pydantic import BaseModel, Field
from typing import Literal, Optional, List

from modules import (
    Ingestion,
    SemanticRouter, 
    StructuralNormalizer, 
    SchemaExtractor,
    SchemaDiscovery,
    PipelineResult
)



# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# --- Sub-models for better organization ---

class PersonalInfo(BaseModel):
    """Information about the taxpayer identifying who matches this document."""
    full_name: Optional[str] = Field(None, description="Full name of the taxpayer found on address block (e.g., JOHN SMITH)")
    address: Optional[str] = Field(None, description="Full mailing address including City, Province, and Postal Code")
    social_insurance_number: Optional[str] = Field(None, description="Social Insurance Number (SIN)")

class DocumentMetadata(BaseModel):
    """Official identifiers used for verification."""
    notice_type: str = Field("Notice of Assessment", description="Type of document header")
    tax_year: int = Field(..., description="The taxation year (e.g., 2022)")
    date_issued: Optional[str] = Field(None, description="Date issued (e.g., May 9, 2023)")
    access_code: Optional[str] = Field(None, description="Access code usually found on the right side (e.g., CT92F9Q6)")
    notice_number: Optional[str] = Field(None, description="Notice number (e.g., 0690208)")

class BalanceDetails(BaseModel):
    """Breakdown of the final financial outcome."""
    final_amount: Optional[float] = Field(None, description="The absolute balance amount usually at the bottom of summary")
    balance_type: Optional[Literal["CR", "DR", "Nil"]] = Field(None, description="CR (Credit/Refund), DR (Debit/Owe), or Nil")
    payment_outcome: Optional[str] = Field(None, description="Context: 'Refund', 'Amount to Pay', or 'No amount'")

class RrspDetails(BaseModel):
    """RRSP info usually found on the last page."""
    deduction_limit_next_year: Optional[float] = Field(None, description="RRSP deduction limit for the NEXT year (e.g., 2023 limit on 2022 NOA)")
    available_contribution_room: Optional[float] = Field(None, description="Available contribution room")

# --- Main Model ---

class TaxAssessment(BaseModel):
    """
    Master Schema for Canada Revenue Agency (CRA) Notice of Assessment.
    Uses nested structures for better data handling.
    """
    # 1. Who and When
    personal_info: PersonalInfo
    document_details: DocumentMetadata

    # 2. Key Line Items (Financials)
    # Using specific Line IDs for accuracy
    line_15000_total_income: Optional[float] = Field(None, description="Line 15000: Total income")
    line_23600_net_income: Optional[float] = Field(None, description="Line 23600: Net income")
    line_26000_taxable_income: Optional[float] = Field(None, description="Line 26000: Taxable income")
    
    # 3. Tax Calculation
    line_35000_total_federal_credits: Optional[float] = Field(None, description="Line 35000: Total federal non-refundable tax credits")
    line_42000_net_federal_tax: Optional[float] = Field(None, description="Line 42000: Net federal tax")
    line_42800_net_provincial_tax: Optional[float] = Field(None, description="Line 42800: Net provincial/territorial tax")
    line_43500_total_payable: Optional[float] = Field(None, description="Line 43500: Total payable")
    line_43700_total_tax_deducted: Optional[float] = Field(None, description="Line 43700: Total income tax deducted")
    line_48200_total_credits: Optional[float] = Field(None, description="Line 48200: Total credits")

    # 4. Result
    balance_summary: BalanceDetails

    # 5. Future Planning (Page 3)
    rrsp_info: RrspDetails


class Invoice(BaseModel):
    """Schema for Invoice documents"""
    invoice_number: str = Field(..., description="Invoice number")
    total_amount: Optional[float] = Field(None, description="Total amount due")
    # ... other fields

# Update the Registry
SCHEMAS = {
    "tax": TaxAssessment,
    "invoice": Invoice,
}


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================
def validate_environment():
    """Check that required environment variables are set."""
    if "GEMINI_API_KEY" not in os.environ:
        print("=" * 60)
        print("WARNING: GEMINI_API_KEY not found in environment variables")
        print("=" * 60)
        
        env_path = os.path.join(os.getcwd(), '.env')
        if os.path.exists(env_path):
            print(f".env file found at: {env_path}")
            with open(env_path, 'r') as f:
                if "GEMINI_API_KEY" in f.read():
                    print("   Key exists in file but may not be loaded correctly.")
                else:
                    print("   GEMINI_API_KEY not found in .env file.")
        else:
            print(f".env file NOT found at: {env_path}")
        
        print("\nPlease set GEMINI_API_KEY in your .env file:")
        print('   GEMINI_API_KEY=your_api_key_here')
        print("=" * 60)
        return False
    
    print("[Config] GEMINI_API_KEY loaded successfully")
    return True


def save_result(result: PipelineResult, output_path: str):
    """Save pipeline result to JSON file."""
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "total_pages": result.total_pages,
            "kept_pages": result.kept_pages,
            "dropped_pages": result.dropped_pages,
        },
        "routing_details": result.routing_details,
        "extracted_data": result.extracted_data,
        "confidence_scores": result.confidence_scores,
        "errors": result.errors
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"[Output] Result saved to: {output_path}")


def run_pipeline(
    pdf_path: str, 
    schema_type: str = "tax", 
    verbose: bool = False,
    output_path: Optional[str] = None,
    use_dynamic_schema: bool = False
) -> PipelineResult:
    """
    Execute the complete document extraction pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        schema_type: Type of document schema to use (ignored if dynamic)
        verbose: Print detailed output
        output_path: Optional path to save JSON result
        use_dynamic_schema: Identify fields automatically using AI
    """

    errors = []
    
    print("\n" + "=" * 60)
    print("DOCUMENT FIELD EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Input: {pdf_path}")
    print(f"Mode: {'DYNAMIC DISCOVERY' if use_dynamic_schema else f'STATIC SCHEMA ({schema_type})'}")
    print("=" * 60 + "\n")
    
    # Get schema (if static)
    schema = None
    if not use_dynamic_schema:
        schema = SCHEMAS.get(schema_type, TaxAssessment)

    
    # =========================================================================
    # STEP 1: INGESTION - Convert PDF to High-Res Images
    # =========================================================================
    print("\n[STEP 1/4] INGESTION & VISION ENCODING")
    print("-" * 40)
    
    try:
        images = Ingestion.process(pdf_path)
        total_pages = len(images)
    except Exception as e:
        return PipelineResult(
            success=False,
            total_pages=0,
            kept_pages=0,
            dropped_pages=0,
            routing_details=[],
            markdown_preview="",
            extracted_data={},
            errors=[f"Ingestion failed: {str(e)}"]
        )
    
    if not images:
        return PipelineResult(
            success=False,
            total_pages=0,
            kept_pages=0,
            dropped_pages=0,
            routing_details=[],
            markdown_preview="",
            extracted_data={},
            errors=["No images could be extracted from PDF"]
        )
    
    # =========================================================================
    # STEP 2: SEMANTIC ROUTING - Filter Important Pages
    # =========================================================================
    print("\n[STEP 2/4] SEMANTIC ROUTING")
    print("-" * 40)
    
    router = SemanticRouter()
    kept_indices, routing_details = router.route(images)
    
    if not kept_indices:
        errors.append("No relevant pages found - all pages classified as noise")
        # Fallback: keep all pages if router filtered everything
        print("[Router] Fallback: Keeping all pages since no relevant pages were identified")
        kept_indices = list(range(len(images)))
    
    selected_images = [images[i] for i in kept_indices]
    
    # =========================================================================
    # STEP 3: STRUCTURAL NORMALIZATION - Convert to Markdown
    # =========================================================================
    print("\n[STEP 3/4] STRUCTURAL NORMALIZATION")
    print("-" * 40)
    
    normalizer = StructuralNormalizer()
    markdown_content = normalizer.normalize(selected_images)
    
    if verbose:
        print("\n--- Markdown Preview (first 1000 chars) ---")
        print(markdown_content[:1000])
        print("--- End Preview ---\n")
    
    # =========================================================================
    # STEP 4: SCHEMA EXTRACTION
    # =========================================================================
    print("\n[STEP 4/4] EXTRACTION")
    print("-" * 40)
    
    if use_dynamic_schema:
        # Phase 4a: Discovery
        print("[Dynamic] Discovering Schema from first 3 pages...")
        discovery_module = SchemaDiscovery()
        # Use selected images (or all if none selected) for discovery
        discovery_images = selected_images if selected_images else images
        dynamic_schema = discovery_module.discover(discovery_images)
        
        # Phase 4b: Extraction
        extractor = SchemaExtractor(schema=dynamic_schema, is_dynamic=True)
    else:
        # Static Extraction
        extractor = SchemaExtractor(schema=schema, is_dynamic=False)
        
    extracted_data, confidence_scores = extractor.extract(markdown_content)

    
    success = bool(extracted_data)
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    
    result = PipelineResult(
        success=success,
        total_pages=total_pages,
        kept_pages=len(kept_indices),
        dropped_pages=total_pages - len(kept_indices),
        routing_details=routing_details,
        markdown_preview=markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
        extracted_data=extracted_data,
        confidence_scores=confidence_scores,
        errors=errors
    )
    
    print(f"Total Pages: {result.total_pages}")
    print(f"Kept Pages: {result.kept_pages}")
    print(f"Dropped Pages: {result.dropped_pages}")
    print(f"Extraction Success: {result.success}")
    
    if result.errors:
        print(f"Warnings: {len(result.errors)}")
        for err in result.errors:
            print(f"   - {err}")
    
    print("\n--- EXTRACTED DATA ---")
    print(json.dumps(result.extracted_data, indent=2, ensure_ascii=False))
    print("----------------------\n")
    
    # Save result to file if output path specified
    if output_path:
        save_result(result, output_path)
    
    return result


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Document Field Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline.py --pdf "NOA 2023.pdf"
    python pipeline.py --pdf "invoice.pdf" --schema invoice
    python pipeline.py --pdf "document.pdf" --verbose
    python pipeline.py --pdf "document.pdf" --output result.json
        """
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--schema",
        choices=list(SCHEMAS.keys()),
        default="tax",
        help="Document schema type (default: tax)"
    )
    parser.add_argument("--dynamic", action="store_true", help="Enable Dynamic Schema Discovery (Ignore --schema)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument("--output", "-o", help="Output JSON file path (default: <pdf_name>_result.json)")
    args = parser.parse_args()

    # Validate file exists
    if not os.path.exists(args.pdf):
        print(f"Error: File not found: {args.pdf}")
        return 1

    # Validate environment
    if not validate_environment():
        return 1

    # Determine output path
    output_path = args.output
    if not output_path:
        # Default: same name as PDF with _result.json suffix
        base_name = os.path.splitext(os.path.basename(args.pdf))[0]
        output_path = f"{base_name}_result.json"

    # Run pipeline
    result = run_pipeline(
        args.pdf, 
        schema_type=args.schema, 
        verbose=args.verbose,
        output_path=output_path,
        use_dynamic_schema=args.dynamic
    )

    
    # Exit code based on success
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
