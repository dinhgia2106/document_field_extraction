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
from typing import Optional

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
class TaxAssessment(BaseModel):
    """
    Schema for Tax Assessment / Notice of Assessment documents.
    
    Designed to capture essential information for:
    - Financial verification (loans, mortgages)
    - Tax compliance tracking
    - Identity confirmation
    - Future financial planning (RRSP limits)
    """
    # === TAXPAYER IDENTIFICATION ===
    tax_year: int = Field(..., description="The taxation year of the assessment (e.g., 2023)")
    taxpayer_name: Optional[str] = Field(None, description="Full name of the taxpayer")
    sin_last_3_digits: Optional[str] = Field(
        None, 
        description="Last 3 digits of Social Insurance Number (e.g., '789' from XXX XXX 789). Used as unique identifier."
    )
    
    # === INCOME ASSESSMENT ===
    total_income: Optional[float] = Field(
        None, 
        description="Line 15000 - Total income before any deductions"
    )
    net_income: Optional[float] = Field(
        None, 
        description="Line 23600 - Net income after basic deductions but before tax. Important for credit assessments."
    )
    taxable_income: Optional[float] = Field(
        None, 
        description="Line 26000 - Taxable income (basis for tax calculation)"
    )
    
    # === TAX BALANCE - CRITICAL FOR CASH FLOW ===
    total_payable: Optional[float] = Field(
        None, 
        description="Line 43500 - Total tax payable (tax obligation before credits/deducted amounts)"
    )
    final_balance: Optional[float] = Field(
        None, 
        description="The bottom-line amount. This is the actual money owed or to be refunded."
    )
    balance_type: Optional[str] = Field(
        None, 
        description="'CR' for Credit (refund due to taxpayer) or 'DR' for Debit (amount owed by taxpayer). Critical for cash flow direction."
    )
    
    # === DOCUMENT METADATA ===
    notice_date: Optional[str] = Field(None, description="Date of notice (YYYY-MM-DD format)")
    notice_id: Optional[str] = Field(None, description="Unique notice/assessment reference number")
    
    # === FUTURE FINANCIAL LIMITS ===
    rrsp_deduction_limit_next_year: Optional[float] = Field(
        None, 
        description="RRSP deduction limit for the following year. Key indicator of future contribution room."
    )


class Invoice(BaseModel):
    """Schema for Invoice documents"""
    invoice_number: str = Field(..., description="Invoice number or ID")
    invoice_date: Optional[str] = Field(None, description="Invoice date (YYYY-MM-DD)")
    vendor_name: Optional[str] = Field(None, description="Name of the vendor/seller")
    buyer_name: Optional[str] = Field(None, description="Name of the buyer/customer")
    subtotal: Optional[float] = Field(None, description="Subtotal before tax")
    tax_amount: Optional[float] = Field(None, description="Tax amount")
    total_amount: Optional[float] = Field(None, description="Total amount due")
    due_date: Optional[str] = Field(None, description="Payment due date")


# Schema registry for different document types
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
        
    extracted_data = extractor.extract(markdown_content)

    
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
