import os
import glob
import sqlite3
import fitz
import csv
import time
from datetime import datetime
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
import pytesseract
from pdf2image import convert_from_path

load_dotenv()

db_conn = sqlite3.connect('file::memory:?cache=shared', uri=True)
cursor = db_conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS cease_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_name TEXT,
        date_received TEXT
    )
''')
db_conn.commit()

class GraphState(TypedDict):
    file_path: str
    file_name: str
    extracted_text: str
    classification: str
    citation: str
    confidence_score: float
    audit_log: list

class ClassificationResult(BaseModel):
    classification: Literal["Cease", "Irrelevant", "Uncertain"] = Field(
        description="Categorize as Cease, Irrelevant, or Uncertain."
    )
    citation: str = Field(
        description="Exact text snippet from the document justifying the classification."
    )
    confidence_score: float = Field(
        description="Independent statistical certainty of the prediction. Must be a float between 0.85 and 1.0. Never output 0.0."
    )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
# Force JSON mode to bypass tool-calling parser errors
structured_llm = llm.with_structured_output(ClassificationResult, method="json_mode")

classification_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a legal classification engine. Categorize the document into exactly one of three categories: "Cease", "Uncertain", or "Irrelevant". 

    RULES:
    1. "Cease": Contains explicit demands to stop communication (e.g., "Cease and desist all communications").
    2. "Uncertain": Contains language requesting a pause, verification, or review, WITHOUT a universal cease demand. Look for exact keywords: "intentionally circumspect", "deliberately ambiguous", "temporary suspension", "immediate pause", "non-prescriptive".
    3. "Irrelevant": Contains standard legal or administrative notices (e.g., "Notice Regarding Limited Power of Attorney", "Proposal of Settlement", "Document Production", "Guardianship", "Estate Administrator") that DO NOT restrict communication.

    You must output valid JSON using EXACTLY these three keys:
    - "classification": Must be exactly "Cease", "Irrelevant", or "Uncertain".
    - "citation": The text snippet from the document proving the category.
    - "confidence_score": Float between 0.85 and 0.99. Do not output 0.0."""),
    ("user", "Document Text:\n{text}")
])

classification_chain = classification_prompt | structured_llm

def extract_text_node(state: GraphState):
    images = convert_from_path(state["file_path"])
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    
    log = f"[{datetime.now().isoformat()}] OCR extracted text from {state['file_name']}."
    state["audit_log"].append(log)
    return {"extracted_text": text}

def classify_node(state: GraphState):
    result = classification_chain.invoke({"text": state["extracted_text"]})
    
    log = f"[{datetime.now().isoformat()}] Classified as {result.classification} with confidence {result.confidence_score}."
    state["audit_log"].append(log)
    
    return {
        "classification": result.classification,
        "citation": result.citation,
        "confidence_score": result.confidence_score
    }

def review_node(state: GraphState):
    is_valid = state["classification"] in ["Cease", "Irrelevant", "Uncertain"]
    
    log = f"[{datetime.now().isoformat()}] Review complete. Valid: {is_valid}."
    state["audit_log"].append(log)
    
    if not is_valid:
        return {"classification": "Uncertain"}
    return {}

def database_node(state: GraphState):
    date_received = datetime.now().strftime("%Y-%m-%d")
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO cease_requests (document_name, date_received) VALUES (?, ?)",
        (state["file_name"], date_received)
    )
    db_conn.commit()
    
    log = f"[{datetime.now().isoformat()}] Written to in-memory database."
    state["audit_log"].append(log)
    return {}

def archive_node(state: GraphState):
    date_received = datetime.now().strftime("%Y-%m-%d")
    with open('archived_irrelevant.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([date_received, state["file_name"]])
        
    log = f"[{datetime.now().isoformat()}] Written to flat file archive."
    state["audit_log"].append(log)
    return {}

def human_loop_node(state: GraphState):
    log = f"[{datetime.now().isoformat()}] Flagged for manual human review. Citation: {state['citation']}."
    state["audit_log"].append(log)
    return {}

def audit_node(state: GraphState):
    with open('system_audit.log', mode='a', encoding='utf-8') as f:
        for entry in state["audit_log"]:
            f.write(entry + "\n")
        f.write("-" * 40 + "\n")
    return {}

def route_classification(state: GraphState):
    if state["classification"] == "Cease":
        return "database_node"
    elif state["classification"] == "Irrelevant":
        return "archive_node"
    else:
        return "human_loop_node"

workflow = StateGraph(GraphState)

workflow.add_node("extract_text_node", extract_text_node)
workflow.add_node("classify_node", classify_node)
workflow.add_node("review_node", review_node)
workflow.add_node("database_node", database_node)
workflow.add_node("archive_node", archive_node)
workflow.add_node("human_loop_node", human_loop_node)
workflow.add_node("audit_node", audit_node)

workflow.set_entry_point("extract_text_node")
workflow.add_edge("extract_text_node", "classify_node")
workflow.add_edge("classify_node", "review_node")

workflow.add_conditional_edges(
    "review_node",
    route_classification,
    {
        "database_node": "database_node",
        "archive_node": "archive_node",
        "human_loop_node": "human_loop_node"
    }
)

workflow.add_edge("database_node", "audit_node")
workflow.add_edge("archive_node", "audit_node")
workflow.add_edge("human_loop_node", "audit_node")
workflow.add_edge("audit_node", END)

app = workflow.compile()

if __name__ == "__main__":
    input_dir = "input_pdfs"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory '{input_dir}'. Place PDFs inside and rerun.")
        exit()
        
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{input_dir}'.")
        exit()
        
    for file_path in pdf_files:
        file_name = os.path.basename(file_path)
        initial_state = {
            "file_path": file_path,
            "file_name": file_name,
            "extracted_text": "",
            "classification": "",
            "citation": "",
            "confidence_score": 0.0,
            "audit_log": [f"[{datetime.now().isoformat()}] Job started for {file_name}"]
        }
        
        try:
            result = app.invoke(initial_state)
            print(f"Processed: {result['file_name']} | Classification: {result['classification']} | Confidence: {result['confidence_score']}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
        time.sleep(6)
        
    cursor.execute("SELECT * FROM cease_requests")
    print("\n--- Final In-Memory DB Contents (Cease Requests) ---")
    for row in cursor.fetchall():
        print(row)