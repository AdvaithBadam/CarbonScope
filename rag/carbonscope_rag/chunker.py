"""Step 2: split pages into overlapping chunks and give each chunk a stable ID."""
import hashlib
import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

logger = logging.getLogger(__name__)


def make_chunk_id(source_file: str, page: int, chunk_index: int, text: str) -> str:
    """Deterministic ID: the same chunk always gets the same ID.

    This is what makes re-running ingestion safe. With random uuids (the notebook
    version) every run added duplicates; with stable IDs, upsert just overwrites.
    """
    raw = f"{source_file}|{page}|{chunk_index}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Number the chunks within each page and attach IDs
    counters = {}
    for chunk in chunks:
        key = (chunk.metadata["source_file"], chunk.metadata["page"])
        index = counters.get(key, 0)
        counters[key] = index + 1
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = make_chunk_id(key[0], key[1], index, chunk.page_content)

    logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
    return chunks
