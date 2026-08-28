"""Distributed ingestion job.

Reads text, Markdown, or JSONL documents with Spark, normalizes them in parallel,
then sends bounded batches to the RAGForge API. Embedding and idempotent indexing
remain centralized so the online and batch paths share exactly the same semantics.
"""

import argparse
import json
import urllib.request

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


def post_partition(rows, api_url: str, batch_size: int):
    batch = []
    for row in rows:
        batch.append({"id": row.id, "text": row.text, "source": row.source, "metadata": {}})
        if len(batch) >= batch_size:
            _post(api_url, batch)
            batch = []
    if batch:
        _post(api_url, batch)


def _post(api_url: str, documents: list[dict]) -> None:
    body = json.dumps({"documents": documents}).encode()
    request = urllib.request.Request(f"{api_url}/v1/documents", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status >= 300:
            raise RuntimeError(f"Ingestion API returned {response.status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Glob or object-store path")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("ragforge-ingestion").getOrCreate()
    schema = T.StructType([T.StructField("id", T.StringType()),
                           T.StructField("text", T.StringType()),
                           T.StructField("source", T.StringType())])
    if args.input.endswith(".jsonl") or args.input.endswith(".json"):
        frame = spark.read.schema(schema).json(args.input)
    else:
        frame = (spark.read.text(args.input, wholetext=True)
                 .withColumnRenamed("value", "text")
                 .withColumn("source", F.input_file_name())
                 .withColumn("id", F.sha2("source", 256)))
    clean = (frame.filter(F.col("text").isNotNull() & (F.length("text") > 0))
             .select("id", F.regexp_replace("text", r"\s+", " ").alias("text"), "source")
             .repartition(args.partitions))
    clean.foreachPartition(lambda rows: post_partition(rows, args.api_url, args.batch_size))
    spark.stop()


if __name__ == "__main__":
    main()
