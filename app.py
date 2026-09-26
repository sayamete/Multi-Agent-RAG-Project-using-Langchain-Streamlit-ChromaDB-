"""
app.py
------
Streamlit UI that connects DIRECTLY to your existing .ipynb notebook —
it does not duplicate your pipeline into a separate module.

How it works:
  1. It reads your .ipynb file (which is just a JSON file under the hood).
  2. It runs every CODE cell in order, inside one shared namespace
     (skipping Jupyter-only lines like `!pip install ...` or `%matplotlib ...`,
     since those aren't valid plain Python and can't be exec()'d).
  3. After all cells have run, that namespace has everything your notebook
     defines — including your `search(query, state)` function and `agents`
     dict — exactly as if you'd run the notebook top to bottom.
  4. The Streamlit chat UI then just calls that same `search()` function.

SETUP — do this before running:
  1. Put this app.py in the SAME folder as your .ipynb file.
  2. Change NOTEBOOK_PATH below to your notebook's actual filename.
  3. Make sure your .env file (with GROQ_API_KEY=...) is in this folder too.
  4. (Recommended) In your notebook, remove or comment out the last test
     cell — the one that does `question = '...'; content = search(...)`.
     Otherwise that test question will run automatically every time the
     app starts (it still works either way, it just costs one extra call).

Run with:
    streamlit run app.py
"""

import json
import streamlit as st

NOTEBOOK_PATH = "code.ipynb"  # <-- CHANGE THIS to your actual .ipynb filename


def load_notebook_namespace(notebook_path: str):
    """Reads the .ipynb file and executes its code cells in one namespace."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    namespace = {}

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        code_lines = []
        for line in source_lines:
            stripped = line.strip()
            # skip Jupyter magic / shell commands (e.g. !pip install, %env) —
            # plain Python exec() cannot run these
            if stripped.startswith("!") or stripped.startswith("%"):
                continue
            code_lines.append(line)

        code = "".join(code_lines)
        if not code.strip():
            continue

        try:
            exec(code, namespace)
        except Exception as e:
            st.warning(f"Skipped a notebook cell due to an error: {e}")

    return namespace


@st.cache_resource(show_spinner="Running your notebook (index build + setup)... this happens once.")
def get_namespace():
    return load_notebook_namespace(NOTEBOOK_PATH)


namespace = get_namespace()
search_fn = namespace.get("search")

st.set_page_config(page_title="Multi-Agent RAG Chat", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Agent RAG Chat")
st.caption(f"Running directly on top of: {NOTEBOOK_PATH}")

if search_fn is None:
    st.error(
        "Couldn't find a `search(query, state)` function inside your notebook. "
        "Make sure the notebook defines one (like your current code does) and "
        "that NOTEBOOK_PATH points to the right file."
    )
    st.stop()

if "state" not in st.session_state:
    st.session_state.state = {"history": []}
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- render past messages ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- chat input ----
query = st.chat_input("Ask about India, do a calculation, or check the weather...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, st.session_state.state = search_fn(
                query=query, state=st.session_state.state
            )
        st.markdown(str(answer))

    st.session_state.messages.append({"role": "assistant", "content": str(answer)})

# ---- sidebar: history + reset ----
with st.sidebar:
    st.header("Conversation history")
    for turn in st.session_state.state["history"]:
        st.markdown(f"**Q:** {turn['query']}")
        st.markdown(f"**A:** {turn['answer']}")
        st.divider()

    if st.button("Clear history"):
        st.session_state.state = {"history": []}
        st.session_state.messages = []
        st.rerun()