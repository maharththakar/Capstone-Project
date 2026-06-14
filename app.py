import streamlit as st
import sqlite3
import pandas as pd
import os
import tempfile
from main import app as graph_app, db_conn

# 1. Premium Page Config (Wide Layout)
st.set_page_config(
    layout="wide",
    page_title="Legal AI Navigator",
    page_icon="⚖️",
    initial_sidebar_state="expanded",
)


def render_audit_trace(audit_log):
    icon_map = {
        "OCR_Extraction": "📄",
        "PII_Sanitization": "🛡️",
        "Classification": "🧠",
        "Algorithmic_Review": "⚖️",
        "Database_Write": "💾",
        "Archive_Write": "🗄️",
        "Human_Queue": "🧑‍⚖️",
    }

    for step in audit_log:
        time_str = step.get("timestamp", "").split("T")[-1][:8]
        action = step.get("action", "Unknown")
        icon = icon_map.get(action, "⚡")

        meta_items = []
        for k, v in step.items():
            if k not in ["timestamp", "action"]:
                if isinstance(v, float):
                    v = round(v, 4)
                meta_items.append(
                    f"<b>{k.replace('_', ' ').title()}</b>: <code>{v}</code>"
                )

        meta_str = " &nbsp;|&nbsp; ".join(meta_items)

        st.markdown(
            f"""
            <div style="margin-bottom: 8px; padding-left: 10px; border-left: 3px solid #555;">
                <span style="font-family: monospace; color: #888; font-size: 0.9em;">[{time_str}]</span> 
                <span style="font-weight: 600; margin-left: 8px;">{icon} {action.replace('_', ' ')}</span><br>
                <div style="margin-left: 30px; font-size: 0.85em; color: #bbb;">{meta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = []

# 2. Enterprise Sidebar
with st.sidebar:
    st.title("⚖️ Legal AI Navigator")
    st.markdown("### Capstone Implementation")
    st.markdown("Automated classification of Cease & Desist directives.")
    st.divider()
    st.markdown("**System Architecture**")
    st.markdown(
        "- 🧠 **LLM:** Groq Llama-3.1-8b\n- 🔄 **Orchestration:** LangGraph\n- 🗄️ **Datastore:** SQLite In-Memory\n- 🕵️ **PII:** Auto-Sanitized"
    )

# 3. Main Dashboard Header
st.title("Cease & Desist Pipeline Dashboard")
st.markdown(
    "Automated ingestion, classification, and CRM routing for inbound legal communications."
)

# 4. Tab Navigation
tab_pipeline, tab_database = st.tabs(
    ["🚀 Batch Ingestion Pipeline", "🗄️ System Datastore & Export"]
)

with tab_pipeline:
    # 5. Layout columns for uploader and batch metrics
    col_upload, col_metrics = st.columns([2, 1])

    with col_upload:
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Drop PDF Legal Notices Here",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button(
                "Execute Multi-Agent Batch Pipeline",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.pipeline_results = []

                # 6. Modern status container for professional execution logging
                with st.status(
                    f"Processing {len(uploaded_files)} documents...", expanded=True
                ) as status:
                    progress_bar = st.progress(0)

                    for idx, uploaded_file in enumerate(uploaded_files):
                        st.write(f"Analyzing: `{uploaded_file.name}`...")
                        file_bytes = uploaded_file.read()

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp_file:
                            tmp_file.write(file_bytes)
                            tmp_path = tmp_file.name

                        initial_state = {
                            "file_path": tmp_path,
                            "file_name": uploaded_file.name,
                            "extracted_text": "",
                            "sanitized_text": "",
                            "classification": "",
                            "citation": "",
                            "confidence_score": 0.0,
                            "audit_log": [],
                        }

                        try:
                            result = graph_app.invoke(initial_state)
                            st.session_state.pipeline_results.append(
                                {
                                    "file_name": uploaded_file.name,
                                    "classification": result["classification"],
                                    "audit_log": result["audit_log"],
                                    "error": None,
                                }
                            )
                        except Exception as e:
                            st.session_state.pipeline_results.append(
                                {"file_name": uploaded_file.name, "error": str(e)}
                            )
                        finally:
                            os.unlink(tmp_path)

                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    status.update(
                        label="Batch Execution Complete!",
                        state="complete",
                        expanded=False,
                    )

    with col_metrics:
        st.subheader("Batch Analytics")
        if st.session_state.pipeline_results:
            total = len(st.session_state.pipeline_results)
            cease_count = sum(
                1
                for r in st.session_state.pipeline_results
                if r.get("classification") == "Cease"
            )
            uncertain_count = sum(
                1
                for r in st.session_state.pipeline_results
                if r.get("classification") == "Uncertain"
            )
            irrelevant_count = sum(
                1
                for r in st.session_state.pipeline_results
                if r.get("classification") == "Irrelevant"
            )

            # 7. Live KPI Scorecards
            m1, m2 = st.columns(2)
            m1.metric("Total Processed", total)
            m2.metric("Cease Directives 🚨", cease_count)

            m3, m4 = st.columns(2)
            m3.metric("Requires Review ⚠️", uncertain_count)
            m4.metric("Irrelevant 🟢", irrelevant_count)
        else:
            st.info("Awaiting execution to generate analytics.")

    st.divider()

    if st.session_state.pipeline_results:
        st.subheader("Execution Trace & Results")
        for res in st.session_state.pipeline_results:
            if res.get("error"):
                st.error(f"Error: {res['file_name']} - {res['error']}")
            else:
                # 8. Actionable Result Cards
                with st.container(border=True):
                    col_result1, col_result2 = st.columns([3, 1])
                    with col_result1:
                        st.markdown(f"#### 📄 {res['file_name']}")

                    with col_result2:
                        if res["classification"] == "Cease":
                            st.error("🚨 CEASE DIRECTIVE")
                        elif res["classification"] == "Uncertain":
                            st.warning("⚠️ REVIEW QUEUE")
                        else:
                            st.success("🟢 IRRELEVANT")

                    with st.expander("View LangGraph Audit Trace"):
                        render_audit_trace(res["audit_log"])

with tab_database:
    st.subheader("High-Priority CRM Routing: Cease Directives")
    try:
        df = pd.read_sql_query("SELECT * FROM cease_requests", db_conn)
        if df.empty:
            st.info("No Cease directives currently in datastore.")
        else:
            col_db1, col_db2 = st.columns([3, 1])
            with col_db1:
                st.metric("Total Intercepted Directives", len(df))
            with col_db2:
                # 9. Enterprise CSV Export Functionality
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name="cease_directives_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.warning("Database connection error or table missing.")
