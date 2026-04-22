"""
NL2Vis — Streamlit Web Application
====================================
Three pages:
  ⚡ Quick Generate   – one-shot NL → viz
  🔄 Iterative Studio – refine with history + code canvas
  ⚙️  Server & Config  – vLLM server control, model selection, LLM backend toggle

Run from project root:
    streamlit run web_vis/app.py
"""

import sys
import os
import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before any project imports
# ---------------------------------------------------------------------------
_WEB_VIS_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _WEB_VIS_DIR.parent.resolve()

# Insert paths in correct order: PROJECT_ROOT first, then WEB_VIS_DIR
# This ensures `import core` from web_vis resolves to web_vis/core first
sys.path.insert(0, str(_PROJECT_ROOT))  # Insert first (goes to index 0)
sys.path.insert(0, str(_WEB_VIS_DIR))   # Insert again at index 0 (pushes PROJECT_ROOT to index 1)

# ---------------------------------------------------------------------------
# Load root core/config.py without name-colliding with web_vis/core
# ---------------------------------------------------------------------------
def _load_root_config():
    spec = importlib.util.spec_from_file_location(
        "_nl2vis_root_config", _PROJECT_ROOT / "core" / "config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

try:
    _RC = _load_root_config()
except Exception:
    _RC = None  # graceful degradation when root config is unavailable

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import streamlit as st
import tempfile
import subprocess
import threading
import datetime
import hashlib
import base64
import glob

from core.chat_manager import ChatManager, ChartExecutionResult  # web_vis/core

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KNOWN_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "Qwen/Qwen3.5-27B-FP8",
    "google/gemma-4-31B-it",
    "Custom...",
]

QUANT_OPTIONS = ["None", "awq", "gptq", "squeezellm", "awq_marlin"]
DTYPE_OPTIONS = ["auto", "float16", "bfloat16"]

_LOG_DIR = _WEB_VIS_DIR / "logs"
_LOG_PATH = str(_LOG_DIR / "app_logs.txt")


# ===========================================================================
# Session state initialisation
# ===========================================================================
def _init_state() -> None:
    defaults: dict = {
        # --- Data files ---
        "csv_files": [],        # full list of resolved absolute CSV paths
        "_upload_tmp": None,    # temp dir for drag-and-drop uploads

        # --- Generation history ---
        # Each entry: {query, code, svg_string, success, error_msg, timestamp}
        "history": [],
        "current_code": "",
        "current_svg": None,

        # --- ChatManager cache ---
        "_cm_key": None,
        "_cm": None,

        # --- vLLM server process ---
        "vllm_proc": None,
        "vllm_logs": [],
        "_vllm_log_thread": None,

        # --- Runtime config overrides ---
        "cfg_use_vllm": getattr(_RC, "USE_VLLM", True),
        "cfg_model": getattr(_RC, "VLLM_MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
        "cfg_custom_model": "",
        "cfg_host": getattr(_RC, "VLLM_HOST", "localhost"),
        "cfg_port": int(getattr(_RC, "VLLM_PORT", 8000)),
        "cfg_max_len": int(getattr(_RC, "VLLM_MAX_MODEL_LEN", 8192)),
        "cfg_gpu_mem": float(getattr(_RC, "VLLM_GPU_MEMORY_UTILIZATION", 0.90)),
        "cfg_quant": getattr(_RC, "VLLM_QUANTIZATION", None) or "None",
        "cfg_dtype": getattr(_RC, "VLLM_DTYPE", "auto"),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===========================================================================
# ChatManager helpers
# ===========================================================================
def _cm_cache_key(csv_files: list) -> str:
    return hashlib.md5(str(sorted(csv_files)).encode()).hexdigest()


def _get_chat_manager(csv_files: list) -> ChatManager:
    """Return a cached ChatManager, recreating only when the file set changes."""
    key = _cm_cache_key(csv_files)
    if st.session_state._cm is None or st.session_state._cm_key != key:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

        try:
            st.session_state._cm = ChatManager(csv_files=csv_files, log_path=_LOG_PATH)
            st.session_state._cm_key = key
        except Exception as e:
            st.error(f"Failed to initialize ChatManager:\n{str(e)}")
            raise

    return st.session_state._cm


def _run_generation(query: str, csv_files: list) -> tuple:
    """Run the full Provider→Generator→Corrector pipeline."""
    cm = _get_chat_manager(csv_files)
    code = cm.start(query)
    result: ChartExecutionResult = cm.execute_to_svg(code)
    return code, result


def _exec_code(code: str, csv_files: list) -> ChartExecutionResult:
    """Execute arbitrary code through execute_to_svg (used by the code canvas)."""
    cm = _get_chat_manager(csv_files)
    return cm.execute_to_svg(code)


# ===========================================================================
# vLLM server management
# ===========================================================================
def _server_alive() -> bool:
    proc = st.session_state.vllm_proc
    return proc is not None and proc.poll() is None


def _resolved_model() -> str:
    m = st.session_state.cfg_model
    return st.session_state.cfg_custom_model.strip() if m == "Custom..." else m


def _log_reader_thread(proc: subprocess.Popen) -> None:
    """Background thread: drain stdout/stderr into the session log list."""
    try:
        for raw in iter(proc.stdout.readline, b""):  # type: ignore[union-attr]
            line = raw.decode("utf-8", errors="replace").rstrip()
            st.session_state.vllm_logs.append(line)
    except Exception:
        pass


def _stop_vllm_server() -> None:
    """Stop the vLLM server process and clean up resources."""
    proc = st.session_state.vllm_proc

    if proc is not None and proc.poll() is None:
        st.info("Stopping vLLM server...")
        proc.terminate()

        try:
            proc.wait(timeout=10)
            st.success("Server stopped gracefully.")
        except subprocess.TimeoutExpired:
            st.warning("Server didn't stop gracefully, forcing termination...")
            proc.kill()
            proc.wait(timeout=5)
            st.warning("Server force-stopped.")

    # Clean up references
    st.session_state.vllm_proc = None
    st.session_state._vllm_log_thread = None
    st.session_state._cm = None  # Force ChatManager recreation


def _start_vllm_server() -> None:
    """Start the vLLM server process."""
    _stop_vllm_server()  # Ensure clean state

    model = _resolved_model()
    if not model:
        st.error("No model specified.")
        return

    quant = st.session_state.cfg_quant
    quant = None if quant == "None" else quant

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", st.session_state.cfg_host,
        "--port", str(st.session_state.cfg_port),
        "--max-model-len", str(st.session_state.cfg_max_len),
        "--gpu-memory-utilization", str(st.session_state.cfg_gpu_mem),
        "--dtype", st.session_state.cfg_dtype,
    ]
    if quant:
        cmd.extend(["--quantization", quant])

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.vllm_logs = [f"[{ts}] Launching: {' '.join(cmd)}"]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(_PROJECT_ROOT),
        )
        st.session_state.vllm_proc = proc

        # Start log reader thread
        t = threading.Thread(target=_log_reader_thread, args=(proc,), daemon=True)
        t.start()
        st.session_state._vllm_log_thread = t

        st.success(f"Server process started (PID {proc.pid}). Monitor logs below.")
        st.info("⏳ Wait ~30-60 seconds for model loading before generating visualizations.")

    except Exception as e:
        st.error(f"Failed to start server: {e}")
        st.session_state.vllm_proc = None


# ===========================================================================
# File helpers
# ===========================================================================
def _add_csv_paths(paths: list) -> int:
    """Add resolved CSV paths to session state, validating files first. Returns count added."""
    existing = set(st.session_state.csv_files)
    added = 0
    errors = []

    for p in paths:
        p = str(Path(p).resolve())

        if not p.lower().endswith(".csv"):
            continue

        if p in existing:
            continue

        # Validate CSV is readable
        try:
            import pandas as pd
            pd.read_csv(p, nrows=1)  # Just read first row to validate
        except Exception as e:
            errors.append(f"{Path(p).name}: {str(e)}")
            continue

        st.session_state.csv_files.append(p)
        existing.add(p)
        added += 1

    if errors:
        st.warning(f"Skipped {len(errors)} invalid CSV files:\n" + "\n".join(errors[:5]))

    if added:
        st.session_state._cm = None  # Invalidate cached ChatManager

    return added


def _save_uploads(uploaded_files) -> list:
    if st.session_state._upload_tmp is None:
        st.session_state._upload_tmp = tempfile.mkdtemp(prefix="nl2vis_uploads_")
    paths = []
    for uf in uploaded_files:
        dest = Path(st.session_state._upload_tmp) / uf.name
        dest.write_bytes(uf.read())
        paths.append(str(dest))
    return paths


# ===========================================================================
# UI helpers
# ===========================================================================
def _now_str() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _render_svg(svg_string: str, height: int = 480) -> None:
    b64 = base64.b64encode(svg_string.encode()).decode()
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'style="width:100%;max-height:{height}px;object-fit:contain;'
        f'background:#fff;border-radius:6px"/>',
        unsafe_allow_html=True,
    )


def _svg_download(svg_string: str, filename: str = "visualization.svg") -> None:
    st.download_button(
        label="⬇ Download SVG",
        data=svg_string.encode(),
        file_name=filename,
        mime="image/svg+xml",
    )


def _history_entry(query: str, code: str, result: ChartExecutionResult) -> dict:
    return {
        "query": query,
        "code": code,
        "svg_string": result.svg_string if result.status else None,
        "success": result.status,
        "error_msg": "".join(result.error_msg) if (not result.status and result.error_msg) else None,
        "timestamp": _now_str(),
    }


def _code_canvas(code: str, key_prefix: str, active_tables: list) -> bool:
    """
    Renders an editable code canvas with a Run button.
    Returns True if the user successfully ran modified code (caller should st.rerun()).
    """
    with st.expander("📝 Code Canvas — edit & run", expanded=False):
        edited = st.text_area(
            "code",
            value=code,
            height=340,
            key=f"{key_prefix}_canvas_edit",
            label_visibility="collapsed",
        )
        col_btn, col_hint = st.columns([1, 5])
        with col_btn:
            run = st.button("▶ Run", key=f"{key_prefix}_canvas_run", type="primary")
        with col_hint:
            if edited.strip() != code.strip():
                st.caption("⚠️ Unsaved edits — click Run to apply")

        if run:
            if not active_tables:
                st.error("No CSV files selected.")
                return False
            with st.spinner("Executing…"):
                result = _exec_code(edited, active_tables)
            if result.status:
                st.session_state.current_code = edited
                st.session_state.current_svg = result.svg_string
                return True
            else:
                err = "".join(result.error_msg) if result.error_msg else "Unknown error"
                st.error(f"Execution error:\n```\n{err}\n```")
    return False


# ===========================================================================
# Sidebar
# ===========================================================================
def _sidebar() -> tuple:
    """
    Renders the persistent sidebar.
    Returns (selected_page: str, active_tables: list[str]).
    """
    with st.sidebar:
        # ── Header + status ─────────────────────────────────────────────────
        st.markdown("## 📊 NL2Vis")

        if st.session_state.cfg_use_vllm:
            dot = "🟢" if _server_alive() else "🔴"
            short = _resolved_model().split("/")[-1] or "—"
            st.caption(f"{dot} **{short}** &nbsp;·&nbsp; port {st.session_state.cfg_port}")
        else:
            st.caption("🔵 Azure / OpenAI mode")

        st.divider()

        # ── Navigation ───────────────────────────────────────────────────────
        page = st.radio(
            "Navigate",
            ["⚡ Quick Generate", "🔄 Iterative Studio", "⚙️ Server & Config"],
            key="_page_nav",
            label_visibility="collapsed",
        )

        st.divider()

        # ── Data Files ───────────────────────────────────────────────────────
        st.markdown("**📂 Data Files**")

        # Local filesystem browser
        with st.expander("Browse filesystem", expanded=not st.session_state.csv_files):
            fs_input = st.text_input(
                "Path or glob",
                placeholder="visEval_dataset/databases/activity_1/*.csv",
                help="Relative to project root. Supports * globs and bare directories.",
                key="_fs_input",
            )
            if st.button("Load CSVs", key="_fs_load", use_container_width=True):
                raw = fs_input.strip()
                if raw:
                    if not os.path.isabs(raw):
                        raw = str(_PROJECT_ROOT / raw)
                    if "*" in raw or "?" in raw:
                        found = glob.glob(raw, recursive=True)
                    elif os.path.isdir(raw):
                        found = glob.glob(os.path.join(raw, "**", "*.csv"), recursive=True)
                    elif os.path.isfile(raw):
                        found = [raw]
                    else:
                        found = []
                    n = _add_csv_paths(found)
                    if n:
                        st.toast(f"Added {n} CSV file(s).", icon="✅")
                    else:
                        st.warning("No new CSV files found at that path.")

        # Drag-and-drop upload
        uploaded = st.file_uploader(
            "Or upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            key="_uploader",
        )
        if uploaded:
            saved = _save_uploads(uploaded)
            n = _add_csv_paths(saved)
            if n:
                st.toast(f"Uploaded {n} CSV file(s).", icon="✅")

        # Loaded file list + active selector
        active_tables: list = []
        if st.session_state.csv_files:
            n_files = len(st.session_state.csv_files)
            st.caption(f"{n_files} file{'s' if n_files != 1 else ''} loaded")

            active_tables = st.multiselect(
                "Active tables",
                options=st.session_state.csv_files,
                default=st.session_state.csv_files,
                format_func=lambda p: Path(p).stem,
                key="_active_tables",
                help="Only selected tables are passed to the generation pipeline.",
            )

            if st.button("🗑 Clear all files", key="_clear_files", use_container_width=True):
                st.session_state.csv_files = []
                st.session_state._cm = None
                st.rerun()
        else:
            st.caption("No files loaded yet.")

    return page, active_tables
    
# ===========================================================================
def page_quick_generate(active_tables: list) -> None:
    st.header("⚡ Quick Generate")
    st.caption("Describe a visualization in plain language and generate it instantly.")

    query = st.text_area(
        "Natural language query",
        placeholder="e.g. Show total sales per region as a horizontal bar chart.",
        height=90,
        key="_quick_query",
    )

    if st.button("Generate", type="primary", key="_quick_gen"):
        if not active_tables:
            st.error("Load and select CSV files in the sidebar first.")
            st.stop()
        if not query.strip():
            st.error("Please enter a query.")
            st.stop()

        with st.spinner("Running pipeline…"):
            try:
                code, result = _run_generation(query.strip(), active_tables)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

        entry = _history_entry(query.strip(), code, result)
        st.session_state.history.append(entry)
        st.session_state.current_code = code
        st.session_state.current_svg = entry["svg_string"]

        if not result.status:
            st.error(f"Generation failed:\n```\n{entry['error_msg']}\n```")

    # Current visualization
    if st.session_state.current_svg:
        _render_svg(st.session_state.current_svg)
        _svg_download(st.session_state.current_svg)
    elif st.session_state.current_code and not st.session_state.current_svg:
        last_fail = next(
            (h for h in reversed(st.session_state.history) if not h["success"]), None
        )
        if last_fail:
            st.error(f"Last run failed:\n```\n{last_fail['error_msg']}\n```")

    st.divider()

    if st.session_state.current_code:
        if _code_canvas(st.session_state.current_code, "quick", active_tables):
            st.rerun()


# ===========================================================================
# Page 2 — Iterative Studio
# ===========================================================================
def page_iterative_studio(active_tables: list) -> None:
    st.header("🔄 Iterative Studio")
    st.caption(
        "Refine your visualization step by step. "
        "Every generation is saved to history so you can restore any previous state."
    )

    # Current visualization
    if st.session_state.current_svg:
        _render_svg(st.session_state.current_svg, height=420)
        _svg_download(st.session_state.current_svg)
    else:
        st.info("No visualization yet — enter a query below to get started.")

    if st.session_state.current_code:
        if _code_canvas(st.session_state.current_code, "studio", active_tables):
            st.rerun()

    st.divider()

    # Query input
    query = st.text_area(
        "Query or refinement instruction",
        placeholder="e.g. 'Use a pie chart instead' or 'Show only the top 10 values'",
        height=80,
        key="_studio_query",
    )

    col_new, col_refine, _ = st.columns([2, 2, 6])
    with col_new:
        do_new = st.button("🆕 New Visualization", key="_studio_new")
    with col_refine:
        do_refine = st.button("🔁 Refine", type="primary", key="_studio_refine")

    def _generate(q: str) -> None:
        if not active_tables:
            st.error("Load and select CSV files in the sidebar first.")
            return
        if not q.strip():
            st.error("Please enter a query.")
            return
        with st.spinner("Generating…"):
            try:
                code, result = _run_generation(q.strip(), active_tables)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return
        entry = _history_entry(q.strip(), code, result)
        st.session_state.history.append(entry)
        st.session_state.current_code = code
        st.session_state.current_svg = entry["svg_string"]
        if not result.status:
            st.error(f"Generation failed:\n```\n{entry['error_msg']}\n```")
        st.rerun()

    if do_new:
        st.session_state.history = []
        st.session_state.current_code = ""
        st.session_state.current_svg = None
        _generate(query)

    if do_refine:
        _generate(query)

    # History panel
    if not st.session_state.history:
        return

    st.divider()
    st.subheader(f"History  ({len(st.session_state.history)} entries)")

    for rev_i, entry in enumerate(reversed(st.session_state.history)):
        real_idx = len(st.session_state.history) - 1 - rev_i
        badge = "✅" if entry["success"] else "❌"
        preview = entry["query"][:70] + ("…" if len(entry["query"]) > 70 else "")
        label = f"{badge}  #{real_idx + 1} · {entry['timestamp']} · {preview}"

        with st.expander(label, expanded=(rev_i == 0)):
            col_viz, col_meta = st.columns([3, 2])
            with col_viz:
                if entry["svg_string"]:
                    _render_svg(entry["svg_string"], height=220)
                else:
                    st.error(entry.get("error_msg") or "Unknown error")
            with col_meta:
                st.markdown(f"**Query:** {entry['query']}")
                st.markdown(f"**Time:** {entry['timestamp']}")
                if st.button("↩ Restore", key=f"_restore_{real_idx}"):
                    st.session_state.current_code = entry["code"]
                    st.session_state.current_svg = entry["svg_string"]
                    st.rerun()
                if entry["svg_string"]:
                    _svg_download(entry["svg_string"], f"viz_{real_idx + 1}.svg")

            with st.expander("Code", expanded=False):
                st.code(entry["code"], language="python")


# ===========================================================================
# Page 3 — Server & Config
# ===========================================================================
def page_server_config() -> None:
    st.header("⚙️ Server & Config")

    tab_server, tab_backend = st.tabs(["vLLM Server", "LLM Backend"])

    # ── Tab: vLLM Server ─────────────────────────────────────────────────────
    with tab_server:
        alive = _server_alive()
        st.markdown(f"**Status:** {'🟢 Running' if alive else '🔴 Stopped'}")
        if alive and st.session_state.vllm_proc:
            st.caption(f"PID {st.session_state.vllm_proc.pid}  ·  {_resolved_model()}")

        st.divider()
        st.subheader("Model")

        cur_model = st.session_state.cfg_model
        model_idx = KNOWN_MODELS.index(cur_model) if cur_model in KNOWN_MODELS else len(KNOWN_MODELS) - 1
        st.session_state.cfg_model = st.selectbox(
            "Select model",
            KNOWN_MODELS,
            index=model_idx,
            key="_srv_model_sel",
        )
        if st.session_state.cfg_model == "Custom...":
            st.session_state.cfg_custom_model = st.text_input(
                "HuggingFace model ID",
                value=st.session_state.cfg_custom_model,
                key="_srv_custom_model",
            )

        with st.expander("Advanced parameters"):
            c1, c2 = st.columns(2)
            with c1:
                st.session_state.cfg_host = st.text_input(
                    "Host", value=st.session_state.cfg_host, key="_srv_host"
                )
                st.session_state.cfg_port = st.number_input(
                    "Port",
                    value=st.session_state.cfg_port,
                    min_value=1024, max_value=65535, step=1,
                    key="_srv_port",
                )
                st.session_state.cfg_max_len = st.number_input(
                    "Max model length",
                    value=st.session_state.cfg_max_len,
                    min_value=512, step=512,
                    key="_srv_maxlen",
                )
            with c2:
                st.session_state.cfg_gpu_mem = st.slider(
                    "GPU memory utilization",
                    0.1, 1.0,
                    value=st.session_state.cfg_gpu_mem,
                    step=0.05,
                    key="_srv_gpu",
                )
                q_idx = QUANT_OPTIONS.index(st.session_state.cfg_quant) if st.session_state.cfg_quant in QUANT_OPTIONS else 0
                st.session_state.cfg_quant = st.selectbox(
                    "Quantization", QUANT_OPTIONS, index=q_idx, key="_srv_quant"
                )
                d_idx = DTYPE_OPTIONS.index(st.session_state.cfg_dtype) if st.session_state.cfg_dtype in DTYPE_OPTIONS else 0
                st.session_state.cfg_dtype = st.selectbox(
                    "Dtype", DTYPE_OPTIONS, index=d_idx, key="_srv_dtype"
                )

        # Start / Stop buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶ Start Server", type="primary", key="_srv_start"):
                _start_vllm_server()
                st.success("Server process launched. Monitor logs below.")
                st.rerun()
        with col_stop:
            if st.button("⏹ Stop Server", key="_srv_stop", disabled=not alive):
                _stop_vllm_server()
                st.info("Server stopped.")
                st.rerun()

        # Copy-friendly shell command
        with st.expander("📋 Equivalent shell command"):
            model = _resolved_model()
            quant_line = f" \\\n  --quantization {st.session_state.cfg_quant}" if st.session_state.cfg_quant != "None" else ""
            shell_cmd = (
                f"python -m vllm.entrypoints.openai.api_server \\\n"
                f"  --model {model} \\\n"
                f"  --host {st.session_state.cfg_host} \\\n"
                f"  --port {st.session_state.cfg_port} \\\n"
                f"  --max-model-len {st.session_state.cfg_max_len} \\\n"
                f"  --gpu-memory-utilization {st.session_state.cfg_gpu_mem} \\\n"
                f"  --dtype {st.session_state.cfg_dtype}"
                f"{quant_line}"
            )
            st.code(shell_cmd, language="bash")

        # Live logs
        st.divider()
        col_ref, col_cnt = st.columns([1, 5])
        with col_ref:
            if st.button("🔄 Refresh Logs", key="_srv_refresh"):
                st.rerun()
        with col_cnt:
            st.caption(f"{len(st.session_state.vllm_logs)} log line(s) captured")

        with st.expander("Server Logs", expanded=bool(st.session_state.vllm_logs)):
            if st.session_state.vllm_logs:
                # Show last 300 lines to avoid giant renders
                st.code("\n".join(st.session_state.vllm_logs[-300:]), language=None)
            else:
                st.caption("No logs yet. Start the server to see output.")

    # ── Tab: LLM Backend ─────────────────────────────────────────────────────
    with tab_backend:
        st.subheader("LLM Backend")

        use_vllm = st.toggle(
            "Use local vLLM server",
            value=st.session_state.cfg_use_vllm,
            key="_be_toggle",
        )
        if use_vllm != st.session_state.cfg_use_vllm:
            st.session_state.cfg_use_vllm = use_vllm
            st.session_state._cm = None

        if use_vllm:
            st.info(
                f"Requests go to `http://{st.session_state.cfg_host}"
                f":{st.session_state.cfg_port}/v1`. "
                "Configure model and server settings in the **vLLM Server** tab."
            )
        else:
            st.info("Using Azure OpenAI / OpenAI API.")
            with st.expander("Azure / OpenAI credentials", expanded=True):
                az_ep = st.text_input(
                    "Azure endpoint",
                    value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                    key="_be_ep",
                )
                az_key = st.text_input(
                    "Azure API key",
                    type="password",
                    value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
                    key="_be_key",
                )
                az_model = st.text_input(
                    "Deployment / model name",
                    value=os.environ.get(
                        "AZURE_MODEL_NAME",
                        getattr(_RC, "AZURE_MODEL_NAME", "gpt-4o"),
                    ),
                    key="_be_model",
                )
                openai_key = st.text_input(
                    "OpenAI API key (optional, for vision)",
                    type="password",
                    value=os.environ.get("OPENAI_API_KEY", ""),
                    key="_be_openai_key",
                )
                if st.button("Apply credentials", key="_be_apply"):
                    errors = []

                    # Validate inputs
                    if az_ep and not az_ep.startswith("https://"):
                        errors.append("Azure endpoint must start with https://")
                    if az_key and len(az_key) < 20:
                        errors.append("Azure API key seems too short (expected 32+ chars)")
                    if not az_model:
                        errors.append("Model/deployment name is required")

                    if errors:
                        st.error("Validation failed:\n" + "\n".join(f"• {e}" for e in errors))
                    else:
                        # Apply to environment
                        if az_ep:
                            os.environ["AZURE_OPENAI_ENDPOINT"] = az_ep
                        if az_key:
                            os.environ["AZURE_OPENAI_API_KEY"] = az_key
                        if az_model:
                            os.environ["AZURE_MODEL_NAME"] = az_model
                        if openai_key:
                            os.environ["OPENAI_API_KEY"] = openai_key

                        # Force ChatManager recreation
                        st.session_state._cm = None
                        st.session_state._cm_key = None

                        st.success("✓ Credentials applied. Next generation will use these settings.")
                        st.info("Note: Credentials are not persisted—set them again on next app restart.")


# ===========================================================================
# Entry point
# ===========================================================================
def main() -> None:
    st.set_page_config(
        page_title="NL2Vis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()

    page, active_tables = _sidebar()

    if page == "⚡ Quick Generate":
        page_quick_generate(active_tables)
    elif page == "🔄 Iterative Studio":
        page_iterative_studio(active_tables)
    elif page == "⚙️ Server & Config":
        page_server_config()


if __name__ == "__main__":
    main()

