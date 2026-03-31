import streamlit as st
import os
import tempfile
from quality_agent import (
    run_pipeline_stepwise, detect_lang, read_code_file,
    ALL_TOOLS, UNIVERSAL_TOOLS, PYTHON_TOOLS, PIPELINE_STEPS,
)

st.set_page_config(page_title="Code Quality Agent", page_icon="Q", layout="wide")

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .step-badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.82em; font-weight: 600; margin-right: 6px;
    }
    .badge-done  { background: #d4edda; color: #155724; }
    .badge-run   { background: #fff3cd; color: #856404; }
    .badge-wait  { background: #e2e3e5; color: #6c757d; }
    .tool-card {
        border: 1px solid #ddd; border-radius: 8px; padding: 10px 14px;
        margin: 4px 0; background: #fafafa;
    }
    .tool-selected { border-color: #28a745; background: #f0fff4; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar: input ──────────────────────────────────────────────────────────
st.sidebar.title("Code Quality Agent")
st.sidebar.caption("v7 — Guidelines-Driven | RAG")

input_mode = st.sidebar.radio("Input method", ["Paste code", "Upload file", "File path"])

code = ""
filename = ""

if input_mode == "Paste code":
    code = st.sidebar.text_area("Paste your code below", height=260, placeholder="def hello():\n    print('hi')")

elif input_mode == "Upload file":
    uploaded = st.sidebar.file_uploader("Upload a source file", type=[
        "py", "js", "ts", "java", "c", "cpp", "h", "cs", "go", "rs",
        "rb", "php", "sql", "r", "vb", "pas", "txt",
    ])
    if uploaded is not None:
        code = uploaded.getvalue().decode("utf-8", errors="replace")
        filename = uploaded.name
        st.sidebar.success(f"Loaded **{filename}** ({len(code.splitlines())} lines)")

elif input_mode == "File path":
    fpath = st.sidebar.text_input("Absolute or relative path to source file")
    if fpath:
        loaded, info = read_code_file(fpath)
        if loaded:
            code = loaded
            filename = info
            st.sidebar.success(f"Loaded **{filename}** ({len(code.splitlines())} lines)")
        else:
            st.sidebar.error(info)

st.sidebar.markdown("---")
analysis_mode = st.sidebar.radio("Analysis mode", ["Auto (agent decides)", "Manual (specify request)"])
user_request = ""
if analysis_mode.startswith("Manual"):
    user_request = st.sidebar.text_input("What should the agent check?", placeholder="e.g. check naming conventions and docstrings")

run_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ── Tool catalogue (always visible) ────────────────────────────────────────
with st.sidebar.expander("Available tools catalogue"):
    st.markdown("**Universal** (all languages)")
    for name, info in UNIVERSAL_TOOLS.items():
        st.markdown(f"- `{name}` — {info['desc']}")
    st.markdown("**Python-only** (AST-based)")
    for name, info in PYTHON_TOOLS.items():
        st.markdown(f"- `{name}` — {info['desc']}")

# ── Main area ───────────────────────────────────────────────────────────────
st.title("Code Quality Agent")

if not run_clicked:
    st.info("Paste code or upload a file in the sidebar, then click **Run Analysis**.")

    # Show pipeline overview
    st.subheader("Pipeline overview")
    cols = st.columns(len(PIPELINE_STEPS))
    for i, (sid, label, _fn) in enumerate(PIPELINE_STEPS):
        with cols[i]:
            st.markdown(f"**{i+1}.** {label}")

    st.subheader("How it works")
    st.markdown("""
1. **Language Detection** — pattern-matching on code keywords to identify the language.
2. **RAG Guidelines Lookup** — retrieves relevant coding style rules from ingested PDF guidelines (ChromaDB + sentence-transformers).
3. **Tool Selection** — the LLM reads the code + guidelines and picks which analysis tools to run (and which guideline checks to perform).
4. **Run Analysis Tools** — the selected static-analysis tools run on the code and produce metrics.
5. **LLM Evaluation** — the LLM interprets tool results against the guidelines and produces a PASS/FAIL verdict.
6. **Report Generation** — everything is assembled into a final quality report.
    """)
    st.stop()

# ── Validation ──────────────────────────────────────────────────────────────
if not code.strip():
    st.warning("Please provide some code to analyse.")
    st.stop()

mode = "auto" if analysis_mode.startswith("Auto") else "manual"

# ── Code preview ────────────────────────────────────────────────────────────
with st.expander("Code preview", expanded=False):
    lang_hint = detect_lang(code)
    st.code(code, language=lang_hint if lang_hint != "unknown" else None, line_numbers=True)

# ── Run pipeline with live progress ─────────────────────────────────────────
st.subheader("Pipeline execution")

# Progress bar
progress = st.progress(0, text="Starting pipeline...")
total_steps = len(PIPELINE_STEPS)

# Step containers (pre-create so they appear in order)
step_containers = []
for i, (_sid, label, _fn) in enumerate(PIPELINE_STEPS):
    step_containers.append(st.container())

final_state = None

for idx, (step_id, label, state) in enumerate(run_pipeline_stepwise(code, mode, user_request)):
    pct = int((idx + 1) / total_steps * 100)
    progress.progress(pct, text=f"Step {idx+1}/{total_steps}: {label}")
    final_state = state

    with step_containers[idx]:
        st.markdown(f'<span class="step-badge badge-done">DONE</span> **Step {idx+1} — {label}**', unsafe_allow_html=True)

        # Per-step details
        if step_id == "node_detect":
            lang = state["detected_language"]
            avail = ALL_TOOLS if lang == "python" else UNIVERSAL_TOOLS
            st.markdown(f"Detected language: **{lang}**")
            st.markdown(f"Tools available for {lang}: **{len(avail)}** ({', '.join(avail.keys())})")

        elif step_id == "node_guidelines":
            guidelines = state["guidelines_context"]
            if guidelines:
                st.success(f"Retrieved guidelines ({len(guidelines)} chars)")
                with st.expander("View retrieved guidelines"):
                    st.text(guidelines[:3000])
            else:
                st.warning("No guidelines found in RAG for this language.")

        elif step_id == "node_pick_tools":
            selected = state["tools_to_run"]
            lang = state["detected_language"]
            avail = ALL_TOOLS if lang == "python" else UNIVERSAL_TOOLS

            st.markdown("**Tool selection by the LLM brain:**")

            # Visual grid showing selected vs not
            tool_cols = st.columns(4)
            for i, (tname, tinfo) in enumerate(avail.items()):
                is_sel = tname in selected
                icon = "**>>>**" if is_sel else ""
                css = "tool-selected" if is_sel else ""
                with tool_cols[i % 4]:
                    st.markdown(
                        f'<div class="tool-card {css}">{icon} <code>{tname}</code><br/>'
                        f'<small>{tinfo["desc"]}</small></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown(f"Selected **{len(selected)}** / {len(avail)} tools: `{'`, `'.join(selected)}`")

            # Guideline checks
            checks = state["guideline_checks"]
            if checks:
                st.markdown("**Guideline checks requested:**")
                st.text(checks)

        elif step_id == "node_run_tools":
            st.markdown(f"Ran **{len(state['tools_to_run'])}** tools.")
            with st.expander("Raw tool results"):
                st.text(state["tool_results"])

        elif step_id == "node_evaluate":
            with st.expander("LLM evaluation (detailed)"):
                st.markdown(state["interpretation"])

        elif step_id == "node_report":
            pass  # shown below

progress.progress(100, text="Pipeline complete!")

# ── Final report ────────────────────────────────────────────────────────────
if final_state:
    st.markdown("---")
    st.subheader("Final Quality Report")

    report = final_state["final_report"]

    # Try to extract verdict
    verdict = "UNKNOWN"
    if "VERDICT: PASS" in report.upper() or "VERDICT:PASS" in report.upper():
        verdict = "PASS"
    elif "VERDICT: FAIL" in report.upper() or "VERDICT:FAIL" in report.upper():
        verdict = "FAIL"

    col1, col2, col3 = st.columns(3)
    col1.metric("Language", final_state["detected_language"])
    col2.metric("Tools run", len(final_state["tools_to_run"]))
    col3.metric("Verdict", verdict)

    st.text(report)

    # Download button
    st.download_button(
        "Download report",
        data=report,
        file_name="quality_report.txt",
        mime="text/plain",
    )
