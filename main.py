import os
import glob
import sqlite3
import csv
import time
import json
import re
from datetime import datetime
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# check_same_thread=False allows Streamlit to interact with the DB
db_conn = sqlite3.connect(
    "file::memory:?cache=shared", uri=True, check_same_thread=False
)
cursor = db_conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS cease_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_name TEXT,
        date_received TEXT
    )
""")
db_conn.commit()


class GraphState(TypedDict):
    file_path: str
    file_name: str
    extracted_text: str
    sanitized_text: str
    classification: str
    citation: str
    confidence_score: float
    audit_log: list


class ClassificationResult(BaseModel):
    classification: Literal["Cease", "Irrelevant", "Uncertain"] = Field(
        description="MUST be 'Cease' if explicit demand. MUST be 'Uncertain' if ambiguous/pause. MUST be 'Irrelevant' for standard notices."
    )
    citation: str = Field(
        description="Exact text snippet justifying the classification."
    )
    confidence_score: float = Field(
        description="Independent statistical certainty of the prediction. Float between 0.85 and 0.99."
    )


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
structured_classifier = llm.with_structured_output(
    ClassificationResult, method="json_mode"
)

classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a multilingual legal classification engine. Categorize the document into exactly one of three categories: "Cease", "Uncertain", or "Irrelevant". 

    RULES:
    1. "Cease": Explicit customer demands to stop communication (e.g., "Cease and desist", "Cese y desista").
    2. "Uncertain": Language requesting a pause, verification, or review. Matches semantic equivalents of: "circumspect", "ambiguous", "temporary suspension", "immediate pause", or "non-prescriptive".
    3. "Irrelevant": Standard enterprise notices without communication restrictions.

    You must output valid JSON containing EXACTLY and ONLY these three keys:
    - "classification": The assigned category.
    - "citation": Exact original text snippet justifying the decision. Do not translate the citation.
    - "confidence_score": Float between 0.85 and 0.99.
    
    Do NOT output 'translation', 'translated_text', or any other unmapped keys.""",
        ),
        ("user", "Document Text:\n{text}"),
    ]
)

classification_chain = classification_prompt | structured_classifier


def extract_text_node(state: GraphState):
    images = convert_from_path(state["file_path"])
    text = "".join([pytesseract.image_to_string(img) for img in images])[:8000]

    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "OCR_Extraction",
            "status": "Success",
        }
    )
    return {"extracted_text": text}


def sanitize_node(state: GraphState):
    text = state["extracted_text"]
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", text)
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL REDACTED]", text
    )
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE REDACTED]", text)

    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "PII_Sanitization",
            "status": "Success",
        }
    )
    return {"sanitized_text": text}


def classify_node(state: GraphState):
    result = classification_chain.invoke({"text": state["sanitized_text"]})

    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "Classification",
            "predicted_category": result.classification,
            "confidence": result.confidence_score,
        }
    )

    return {
        "classification": result.classification,
        "citation": result.citation,
        "confidence_score": result.confidence_score,
    }


def review_node(state: GraphState):
    is_valid = state["classification"] in ["Cease", "Irrelevant", "Uncertain"]
    final_class = state["classification"] if is_valid else "Uncertain"

    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "Algorithmic_Review",
            "is_valid": is_valid,
            "final_category": final_class,
        }
    )

    return {"classification": final_class}


def database_node(state: GraphState):
    date_received = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "INSERT INTO cease_requests (document_name, date_received) VALUES (?, ?)",
        (state["file_name"], date_received),
    )
    db_conn.commit()
    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "Database_Write",
            "status": "Success",
        }
    )
    return {}


def archive_node(state: GraphState):
    date_received = datetime.now().strftime("%Y-%m-%d")
    with open(
        "archived_irrelevant.csv", mode="a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([date_received, state["file_name"]])
    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "Archive_Write",
            "status": "Success",
        }
    )
    return {}


def human_loop_node(state: GraphState):
    state["audit_log"].append(
        {
            "timestamp": datetime.now().isoformat(),
            "action": "Human_Queue",
            "status": "Flagged",
        }
    )
    return {}


def audit_node(state: GraphState):
    with open("system_audit.jsonl", mode="a", encoding="utf-8") as f:
        log_entry = {
            "document": state["file_name"],
            "final_classification": state["classification"],
            "trace": state["audit_log"],
        }
        f.write(json.dumps(log_entry) + "\n")
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
workflow.add_node("sanitize_node", sanitize_node)
workflow.add_node("classify_node", classify_node)
workflow.add_node("review_node", review_node)
workflow.add_node("database_node", database_node)
workflow.add_node("archive_node", archive_node)
workflow.add_node("human_loop_node", human_loop_node)
workflow.add_node("audit_node", audit_node)

workflow.set_entry_point("extract_text_node")
workflow.add_edge("extract_text_node", "sanitize_node")
workflow.add_edge("sanitize_node", "classify_node")
workflow.add_edge("classify_node", "review_node")

workflow.add_conditional_edges(
    "review_node",
    route_classification,
    {
        "database_node": "database_node",
        "archive_node": "archive_node",
        "human_loop_node": "human_loop_node",
    },
)

workflow.add_edge("database_node", "audit_node")
workflow.add_edge("archive_node", "audit_node")
workflow.add_edge("human_loop_node", "audit_node")
workflow.add_edge("audit_node", END)

app = workflow.compile()
