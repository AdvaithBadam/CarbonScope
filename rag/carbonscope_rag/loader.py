"""Step 1: read PDFs into LangChain Documents (one Document per page)."""
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf(path: Path) -> List[Document]:
    """Load one PDF. Page numbers are 1-based so citations match what a reader sees."""
    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning("%s page %d has no extractable text (scanned?) - skipped", path.name, page_number)
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source_file": path.name, "page": page_number, "total_pages": total_pages},
            )
        )
    return documents


def load_all_pdfs(data_dir: Path, strict: bool = False) -> List[Document]:
    """Load every PDF found under data_dir (recursively).

    Args:
        data_dir: Directory to search for PDF files.
        strict: When True, re-raise any PDF load error instead of logging and
                continuing. Use strict=True in production ingestion scripts so
                a corrupt file fails loudly rather than silently producing an
                incomplete index.
    """
    pdf_files = sorted(Path(data_dir).glob("**/*.pdf"))
    logger.info("Found %d PDF files in %s", len(pdf_files), data_dir)

    all_documents: List[Document] = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(pdf_file)
            logger.info("Loaded %d pages from %s", len(docs), pdf_file.name)
            all_documents.extend(docs)
        except Exception:
            if strict:
                raise
            # logger.exception keeps the full traceback, unlike a bare print(e)
            logger.exception("Failed to load %s", pdf_file.name)

    logger.info("Total pages loaded: %d", len(all_documents))
    return all_documents
