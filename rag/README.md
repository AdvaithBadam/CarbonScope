# CarbonScope RAG

Answers questions about the CCTS documents, with page citations.

## Setup
```powershell
pip install -r rag/requirements.txt
Copy-Item .env.example .env   # then put your GROQ_API_KEY in .env (never commit it)
```
Put the PDFs in `rag/data/pdf/`, then from the `rag/` folder:

```bash
python scripts/ingest.py                      # build the index (safe to re-run)
python scripts/ask.py "Explain the project registration phase" --show-context
pytest tests                                  # offline tests, no key needed
```

## Using it from the backend
```python
from carbonscope_rag import RAGPipeline

rag = RAGPipeline()            # create ONCE at startup, reuse for every request
result = rag.query("What is the validation step?", history=[...])
# {"answer": str, "sources": [{n, source_file, page, score, preview}], "confidence": float, "grounded": bool}

for event in rag.stream("..."):   # for token streaming: sources -> token... -> done
    ...
```

## Files
| File | Job |
|---|---|
| `config.py` | all settings (paths, models, chunk size, `MIN_SCORE`) |
| `loader.py` | PDF -> one Document per page (1-based page numbers) |
| `chunker.py` | split into chunks, stable chunk IDs |
| `embeddings.py` | SentenceTransformer wrapper, normalised vectors |
| `vector_store.py` | ChromaDB: upsert, delete by source, cosine query |
| `retriever.py` | query -> chunks above `MIN_SCORE` |
| `prompts.py` | system prompt + context formatting |
| `pipeline.py` | `RAGPipeline.query()` / `.stream()` |

## .gitignore
`.env`, `.venv/`, `__pycache__/`, and `rag/data/vector_store/` are all covered by the root `.gitignore`.
