# Legal PDF Classification Multi-Agent System

An automated pipeline for ingesting, extracting, and classifying legal PDF communications (Cease & Desist directives vs. standard notices) using a LangGraph multi-agent architecture and LLM inference.

## Architecture
* **Orchestration:** LangGraph state machine.
* **Extraction Agent:** Optical Character Recognition (OCR) via Tesseract/pdf2image to process rasterized PDF scans.
* **Classification Agent:** Llama-3.1-8b via Groq API. Utilizes strict Pydantic JSON schema constraints and keyword anchoring to prevent hallucination.
* **Review Agent:** Algorithmic validation node to verify schema adherence.
* **Routing Agents:**
  * **Database Agent:** Writes `Cease` directives to SQLite.
  * **Archiving Agent:** Appends `Irrelevant` notices to a CSV flat file.
  * **Human-in-the-Loop Node:** Queues `Uncertain` (ambiguous legal framing) documents for manual review.
* **Audit Agent:** Logs state transitions, citations, and metadata to `system_audit.jsonl`.

## Setup
1. Install system dependencies:
   `brew install tesseract poppler`
2. Create virtual environment:
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Install Python packages:
   `pip install -r requirements.txt`
4. Set Groq API key in `.env`:
   `GROQ_API_KEY="your_key"`

## Execution
1. Place target PDFs in the `/input_pdfs` directory.
2. Execute the graph:
   `python3 main.py`

## Output Artifacts
* **SQLite In-Memory DB:** Contains ingested `Cease` records (dumped to console upon completion).
* **archived_irrelevant.csv:** Appended list of processed administrative notices.
* **system_audit.jsonl:** Immutable JSON log of all classification logic, specific text citations, and routing decisions for compliance.
