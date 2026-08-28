"""Run RAGAS faithfulness and answer-relevancy metrics against a JSONL dataset."""

import argparse
import json
import urllib.request

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness


def ask(api_url: str, question: str) -> dict:
    request = urllib.request.Request(
        f"{api_url}/v1/ask", data=json.dumps({"question": question}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSONL rows: question, ground_truth")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default="reports/ragas.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.dataset) if line.strip()]
    samples = []
    for row in rows:
        result = ask(args.api_url, row["question"])
        samples.append({"question": row["question"], "answer": result["answer"],
                        "contexts": [c["excerpt"] for c in result["citations"]],
                        "ground_truth": row["ground_truth"]})
    scores = evaluate(Dataset.from_list(samples), metrics=[faithfulness, answer_relevancy])
    with open(args.output, "w") as output:
        json.dump(scores.to_pandas().to_dict(orient="records"), output, indent=2)


if __name__ == "__main__":
    main()

