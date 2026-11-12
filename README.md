# ClipRank Labs Knowledge Assistant — Module 08 RAG Challenge

A multi-document Retrieval-Augmented Generation chatbot for the made-up company **ClipRank Labs**, maker of the SEO Clip Score Tool. Support agents and customers ask questions and get answers with the source file name and a quote. All data is fictional.

Demo video (3-5 minutes): **PASTE YOUR VIDEO LINK HERE**

## Start here

Install the packages, add one key to `.env`, build the index, then start the chatbot. Searching runs on your own computer. **Only the final answer needs a Groq or Gemini key.**

## Contents

- `streamlit_app.py` — the chatbot page (run locally)
- `rag_pipeline.py` — load, split, embed, Chroma, search, ask the AI, check sources
- `build_index.py` — rebuild the index: `python build_index.py [chunk_size] [overlap]`
- `run_tests.py` — runs the 20 tests and writes `test_results.csv`
- `experiments.py` — k = 1/3/5 and chunk-setting comparison, writes `experiment_results.csv`
- `eval_utils.py` — small helpers for the two scripts above
- `knowledge_base/` — 6 fictional documents (overview, products, current policy v3.0, archived policy v2.0, operations/FAQs, release notes with a prompt-injection test)
- `tests/test_questions.csv` — 20 tests with expected answers and sources, written before testing
- `docs/requirements_checklist.md` — where each knowledge-base requirement is met
- `.streamlit/config.toml` — the green, beige and white colours
- `requirements.txt` — Python packages
- `.env.example` — example credential names; **never publish real API keys**

## Option A — VS Code (Python 3.11 or 3.12)

1. Extract the ZIP, then **open the `cliprank-rag` folder** in VS Code.
2. In the VS Code terminal run `python -m venv .venv`, then `.venv\Scripts\Activate.ps1`, then `python -m pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`, add either `GROQ_API_KEY` or `GOOGLE_API_KEY`, and set `LLM_PROVIDER` (`groq` or `gemini`).
4. Run `python build_index.py` once. The first run downloads the small Hugging Face embedding model.
5. Run `python -m streamlit run streamlit_app.py` and open the localhost link.

## How the answers stay trustworthy

- **Details on every chunk:** `filename`, `doc_id`, `version`, `status`, `effective_date`.
- **Policy version rule (in code, not in the prompt):** `detect_policy_mode()` reads the question. No word about old rules gives `current` (Chroma filter `status = current`). Only old words give `archived`. Old words plus current/change words give `compare` (k current chunks plus k archived chunks).
- **Refusal:** if the evidence is not enough, the AI must reply with exactly "I could not find enough information in the provided documents." and no sources.
- **Sources:** the AI returns a file name and a quote. `verify_sources()` checks that the quote really appears in a retrieved chunk, and the app shows a warning if not.
- **Prompt injection:** retrieved text is wrapped in `<retrieved_documents>` and the prompt says it is untrusted data. This is not proof that it works. Run `run_tests.py` (tests 19 and 20) and read the real answers.
- **Conflicts:** if two documents disagree and nothing says which one applies, the AI must explain the conflict.

## Testing and experiments

```powershell
python run_tests.py       # writes test_results.csv; then fill in pass_fail and reason yourself
python experiments.py     # writes experiment_results.csv; then fill in answer_correct and notes
```

Free Groq/Gemini tiers have limits, so both scripts wait between calls (`--sleep 10` waits longer). `python experiments.py --retrieval-only` needs no key. After a run, check that no row says `ERROR`.

## Final live challenge: why editing a file is not enough

Chroma keeps the vectors that were made when the index was built. Editing a document does not change them, so the assistant would still answer from the old text. After editing, rebuild: click **Rebuild index** in the app, or run `python build_index.py`. The rebuild deletes the old collection first, so no old chunks stay.

## Note

Free Groq/Gemini tiers have changing quotas and model availability. The default model names are examples; change them in `.env` if your account requires. `gemini-2.5-flash` is retired for new accounts, so the default Gemini model here is `gemini-3.8-flash`. The embedding model downloads once and needs no Hugging Face API key. The Chroma data is saved in `.rag_store/` and is not part of the ZIP.

## Safety

Prompt construction is a mitigation, not a foolproof prompt-injection defense. Similarity search always returns top matches, even for irrelevant questions, so human review and more advanced retrieval filtering are needed before real-world deployment. Version selection uses keywords, so unusual wording can pick the wrong mode. This is a classroom prototype: real use would also need login and access control, a privacy review, tests on real questions, logging, and a document update process.
