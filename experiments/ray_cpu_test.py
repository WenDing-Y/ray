import argparse
import hashlib
import os
import time

import ray
import ray.data
from ray.data import DataContext


def generate_data(num_rows):
    return [
        {
            "id": i,
            "text": f"Ray Data CPU bottleneck test text {i} " * 10,
        }
        for i in range(num_rows)
    ]


class CpuHeavyTokenizer:
    def __init__(self):
        print(f"Tokenizer initialized, PID={os.getpid()}")

    def __call__(self, batch):
        results = []
        for text in batch["text"]:
            data = str(text).encode("utf-8")
            for _ in range(3000):
                data = hashlib.sha256(data).digest()
            results.append(data.hex())
        return {"token_result": results}


def run(args):
    ray.init(address=args.address)
    print("Cluster resources:", ray.cluster_resources())
    print("Available resources:", ray.available_resources())

    ctx = DataContext.get_current()
    ctx.target_max_block_size = 32 * 1024 * 1024

    print("Generating data...")
    data = generate_data(args.rows)
    ds = ray.data.from_items(data, override_num_blocks=args.blocks)
    del data
    print("Initial blocks:", ds.num_blocks())

    ds = ds.map_batches(
        CpuHeavyTokenizer,
        batch_size=args.batch_size,
        batch_format="numpy",
        concurrency=args.concurrency,
        num_cpus=1,
    )

    print("Starting CPU-intensive pipeline...")
    start = time.perf_counter()
    # 使用写出消费结果，确保执行用户函数。
    ds.write_parquet(args.output)
    elapsed = time.perf_counter() - start

    print("\n========== RESULT ==========")
    print(f"Rows: {args.rows}")
    print(f"Requested blocks: {args.blocks}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Throughput: {args.rows / elapsed:.2f} rows/s")
    print("\n========== DATASET STATS ==========")
    print(ds.stats())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--blocks", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--output", default="/tmp/ray_cpu_output")
    run(parser.parse_args())
