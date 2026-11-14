"""Build (or refresh) the vector store from the PDFs in rag/data/pdf.

    python scripts/ingest.py            # safe to re-run, no duplicates
    python scripts/ingest.py --reset    # wipe the collection first
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carbonscope_rag import config  # noqa: E402
from carbonscope_rag.chunker import split_documents  # noqa: E402
from carbonscope_rag.embeddings import EmbeddingManager  # noqa: E402
from carbonscope_rag.loader import load_all_pdfs  # noqa: E402
from carbonscope_rag.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store")
    parser.add_argument("--reset", action="store_true", help="delete the collection before ingesting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("ingest")

    documents = load_all_pdfs(config.DATA_DIR, strict=True)
    if not documents:
        log.error("No PDF text found in %s", config.DATA_DIR)
        return 1

    chunks = split_documents(documents)
    store = VectorStore()
    if args.reset:
        store.reset()

    # Replace each source file's chunks, so an edited PDF leaves nothing stale behind
    for source_file in sorted({c.metadata["source_file"] for c in chunks}):
        store.delete_source(source_file)

    embeddings = EmbeddingManager().embed_documents([c.page_content for c in chunks])
    store.upsert(chunks, embeddings)
    log.info("Done. %d chunks in collection '%s'.", store.count(), config.COLLECTION_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
