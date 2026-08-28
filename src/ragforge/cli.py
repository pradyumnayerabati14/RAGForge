import argparse
import json
from pathlib import Path

from ragforge.container import get_container
from ragforge.models import IngestDocument


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragforge")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Ingest .txt and .md files")
    ingest.add_argument("path", type=Path)
    ask = sub.add_parser("ask", help="Ask a question in this process")
    ask.add_argument("question")
    args = parser.parse_args()
    container = get_container()
    if args.command == "ingest":
        paths = [args.path] if args.path.is_file() else list(args.path.rglob("*"))
        docs = [IngestDocument(id=str(p.resolve()), text=p.read_text(), source=str(p))
                for p in paths if p.suffix.lower() in {".txt", ".md"}]
        print(json.dumps(vars(container.ingestion.ingest(docs)), indent=2))
    else:
        print(container.qa.ask(args.question).model_dump_json(indent=2))

