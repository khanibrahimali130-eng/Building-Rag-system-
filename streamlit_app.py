"""ClipRank Labs Knowledge Assistant: the Streamlit chat page.

Run:  python -m streamlit run streamlit_app.py
Colours (green, beige, white) are set in .streamlit/config.toml.
The small style block below only adds the banner and the cards.
"""
import html
import os

import streamlit as st

from rag_pipeline import (ask, build_index, current_provider, default_model, get_index,
                          get_llm, is_refusal, DATA_DIR, DEFAULT_K, KEY_NAMES)

st.set_page_config(page_title="ClipRank Labs Knowledge Assistant", page_icon="📚", layout="wide")

# A little CSS for the green banner, the beige source cards and the mode label.
st.markdown("""
<style>
.hero { background: #2F6B4F; padding: 1.4rem 2rem; border-radius: 14px; margin-bottom: 0.8rem; }
.hero h1 { color: #FFFFFF; margin: 0; padding: 0; font-size: 2rem; }
.mode-label { display: inline-block; background: #F5EFE0; color: #2F6B4F; border: 1px solid #2F6B4F;
              padding: 0.15rem 0.7rem; border-radius: 999px; font-size: 0.85rem; margin-bottom: 0.6rem; }
.source-card { background: #F5EFE0; border-left: 5px solid #2F6B4F; padding: 0.7rem 1rem;
               border-radius: 8px; margin-bottom: 0.6rem; }
.source-card .file { color: #2F6B4F; font-weight: 600; }
.source-card .warn { color: #8A5A00; font-size: 0.85rem; margin-top: 0.3rem; }
.note { background: #F5EFE0; border-left: 5px solid #2F6B4F; padding: 0.8rem 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Page title, caption and a short introduction.
st.markdown('<div class="hero"><h1>📚 ClipRank Labs Knowledge Assistant</h1></div>', unsafe_allow_html=True)
st.caption("Module 08 | Local Hugging Face embeddings + Chroma + Groq/Gemini")
st.markdown('<div class="note">Ask about plans and prices, features, refunds, data retention and support steps. '
            'If a fact is missing, the assistant should say so instead of guessing. '
            'Classroom prototype with made-up data.</div>', unsafe_allow_html=True)

MODE_TEXT = {"current": "Current policy only",
             "archived": "Old (archived) policy",
             "compare": "Old and current policy compared"}

# ------------------------------------------------------------ SIDEBAR
with st.sidebar:
    st.header("Classroom settings")
    provider_name = st.selectbox("Generation provider", ["Groq", "Gemini"],
                                 index=0 if current_provider() == "groq" else 1)
    provider = provider_name.lower()
    # The key makes the model box start fresh when you switch provider.
    model_name = st.text_input("LLM model name", default_model(provider), key=f"model_{provider}")
    top_k = st.slider("How many chunks to retrieve?", min_value=1, max_value=5, value=DEFAULT_K)
    # Use the key from .env, or let the user type one for this session only.
    key_name = KEY_NAMES[provider]
    api_key = os.getenv(key_name, "") or st.text_input(f"{key_name} (if not in .env)", type="password")
    st.caption('Current rules are used by default. Say "old" or "previous" to see the archived rules.')

    st.subheader("Changed a document?")
    st.caption("Editing a file does not update the saved vectors. Rebuild the index.")
    rebuild = st.button("Rebuild index")

    st.subheader("Try asking")
    st.markdown("- What is the monthly price of the Pro plan?\n"
                "- I cancel a Business annual plan after 20 days. How much do I get back?\n"
                "- What was the refund window under the old policy, and what is it now?\n"
                "- Does ClipRank Labs offer a public API?")
    st.divider()
    st.caption("The knowledge base is fictional teaching material. Source text is shown with each reply.")


# ------------------------------------------------------------ CACHED HELPERS
@st.cache_resource(show_spinner="Preparing the searchable knowledge base...")
def load_index():
    # Cached, so the index is opened once and not on every click.
    return get_index()


@st.cache_resource(show_spinner="Connecting to the language model...")
def load_llm(provider, model_name, api_key):
    return get_llm(provider, model_name, api_key)


def show_reply(entry):
    """Draw one assistant reply: mode label, answer, sources and the retrieved chunks."""
    st.markdown(f'<span class="mode-label">{MODE_TEXT[entry["mode"]]}</span>', unsafe_allow_html=True)

    if is_refusal(entry["text"]):
        st.markdown(f'<div class="note">{html.escape(entry["text"])}</div>', unsafe_allow_html=True)
    else:
        st.write(entry["text"])

    for s in entry["sources"]:
        warning = ""
        if not s["verified"]:
            warning = '<div class="warn">This quote could not be matched word for word in the retrieved text.</div>'
        st.markdown(f'<div class="source-card"><div class="file">{html.escape(s["filename"])}</div>'
                    f'{html.escape(s["excerpt"])}{warning}</div>', unsafe_allow_html=True)

    with st.expander("Read retrieved chunks"):
        for number, chunk in enumerate(entry["evidence"], start=1):
            m = chunk.metadata
            st.markdown(f'**SOURCE {number} — {m.get("filename", "unknown")}** | version {m.get("version", "n/a")} | '
                        f'{m.get("status", "n/a")} | distance {m.get("distance", "n/a")}')
            st.code(chunk.page_content, language=None)


# ------------------------------------------------------------ CHECK THE DOCUMENTS
if not any(p.suffix.lower() in {".md", ".txt"} for p in DATA_DIR.glob("*")):
    st.error(f"No documents found in {DATA_DIR}. Put the .md or .txt files there.")
    st.stop()

# ------------------------------------------------------------ REBUILD BUTTON
if rebuild:
    try:
        with st.spinner("Rebuilding the index..."):
            _, count = build_index()
        st.cache_resource.clear()
        st.success(f"Done. The index was rebuilt from the documents ({count} chunks).")
    except Exception as error:
        st.error("The index could not be rebuilt.")
        with st.expander("Technical details"):
            st.code(str(error))

try:
    store = load_index()
except Exception as error:
    st.error(f"Could not prepare the index: {error}")
    st.stop()

# ------------------------------------------------------------ CHAT
# Session state keeps the chat visible when Streamlit reruns the page.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        if entry["role"] == "user":
            st.write(entry["text"])
        else:
            show_reply(entry)

question = st.chat_input("Example: What is the monthly price of the Pro plan?")
if question:
    st.session_state.chat_history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)

    # Searching needs no key. Only the answer needs one.
    if not api_key:
        st.warning(f"Add {key_name} in .env or the sidebar to generate an answer.")
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching source chunks and generating answer..."):
                    llm = load_llm(provider, model_name, api_key)
                    result = ask(question.strip(), store, top_k, llm=llm)
            except Exception as error:
                # Quota, network and key errors are possible on free tiers.
                st.error(f"The LLM request failed: {type(error).__name__}. "
                         "Check the key, model, quota, and internet connection.")
                with st.expander("Technical details"):
                    st.code(str(error))
            else:
                entry = {"role": "assistant", "text": result["answer"], "mode": result["mode"],
                         "sources": result["sources"], "evidence": result["chunks"]}
                show_reply(entry)
                st.session_state.chat_history.append(entry)
