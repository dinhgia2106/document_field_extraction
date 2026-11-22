# Document Field Extraction with LLMs

Turns multi-page PDF documents into validated, structured JSON with a vision-LLM pipeline — and compares practical ways to attach a **confidence score** to every extracted field.

Demonstrated on Canadian *Notice of Assessment* tax documents (specimen data in [`samples/`](samples/)); an invoice schema and dynamic schema discovery are also included.

## Pipeline

```
PDF ─► 1. Ingestion ─► 2. Semantic router ─► 3. Structural normalizer ─► 4. Schema extractor ─► JSON
       pages → images   keep A/B pages,       page images → Markdown     Pydantic schema +
                        drop C pages                                     self-correction
```

| Module | What it does |
|---|---|
| `Ingestion` | Renders PDF pages to high-resolution images (`pdf2image`). |
| `SemanticRouter` | Classifies every page as A / B / C by usefulness and drops C pages to save tokens. |
| `StructuralNormalizer` | Transcribes kept pages to Markdown, preserving tables. |
| `SchemaDiscovery` | Optional: proposes a schema for unknown document types (`--dynamic`). |
| `SchemaExtractor` | Extracts fields into a Pydantic model and retries on validation errors. |

Models: Gemini 2.0 Flash (routing, transcription, extraction) and GPT-4o-mini (logprob confidence).

## Field confidence: what was tried

LLM APIs do not return a confidence per field, so four strategies were compared ([full report](docs/experiment-report.md), [method survey](docs/confidence-methods.md)):

| Strategy | Idea | Finding |
|---|---|---|
| Token log-probabilities | Average probability of the value tokens after each key | Works on Gemini 2.0 Flash and GPT-4o; scores skew very high for extraction tasks |
| LLM-as-a-judge | A second model verifies each field against the source | Useful as a verification step |
| Multi-model consensus | 4 Gemini/GPT models vote per field | Most robust, highest cost |
| Self-evaluation | The extractor rates its own output | Least reliable |

## Usage

```bash
pip install -r requirements.txt      # pdf2image also needs Poppler installed
cp .env.example .env                 # add GEMINI_API_KEY and OPENAI_API_KEY

python pipeline.py --pdf samples/noa-2023-specimen.pdf
python pipeline.py --pdf invoice.pdf --schema invoice
python pipeline.py --pdf document.pdf --dynamic --confidence -o result.json
```

## Layout

```
pipeline.py      CLI and output schemas (TaxAssessment, Invoice)
modules.py       the four pipeline modules
samples/         specimen PDFs and their extracted JSON
docs/            experiment report and confidence-method survey
experiments/     logprob support checks for Gemini and OpenAI models
```
