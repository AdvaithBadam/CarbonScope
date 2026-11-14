"""Ask the RAG a question from the terminal.

    python scripts/ask.py "Explain the project registration phase"
    python scripts/ask.py "..." --show-context --min-score 0.3
    python scripts/ask.py "..." --stream
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carbonscope_rag import config  # noqa: E402
from carbonscope_rag.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    parser.add_argument("--show-context", action="store_true", help="print retrieved chunk scores")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.show_context else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    rag = RAGPipeline()

    if args.stream:
        sources = []
        for event in rag.stream(args.question, top_k=args.top_k, min_score=args.min_score):
            if event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "token":
                print(event["text"], end="", flush=True)
        print()
    else:
        result = rag.query(args.question, top_k=args.top_k, min_score=args.min_score)
        print(result["answer"])
        sources = result["sources"]

    if sources:
        print("\nSources:")
        for s in sources:
            print(f"  [{s['n']}] {s['source_file']} (page {s['page']}, score {s['score']})")
            if args.show_context:
                print(f"      {s['preview']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
