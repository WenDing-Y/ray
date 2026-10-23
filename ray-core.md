# Ray Core

## 第一题：Task 和 Actor 有什么区别？

**Task 是一次函数执行，Actor 是可反复调用的有状态实例，Worker 是承载它们的进程。** Task 结束后，Worker 通常可以复用。

### 一、核心区别

| 对比 | Task（普通远程函数任务） | Actor |
| --- | --- | --- |
| 定义 | `@ray.remote` 修饰函数 | `@ray.remote` 修饰类 |
| 调用 | `func.remote()` | `actor.method.remote()` |
| 状态 | 不保证跨调用保留状态 | 实例存活期间保留状态 |
| 进程 | Worker 通常可复用 | 通常由专属 Worker 承载 |
| 并发 | 多个 Task 可并行 | 默认同步 Actor 的方法串行执行 |
| 用途 | 独立计算、数据清洗 | 模型复用、状态管理 |

Actor 方法调用也是 Task，称为 **Actor Task**；远程函数调用称为 **Normal Task**。常见的单返回值调用返回 `ObjectRef`，创建 Actor 则返回 `ActorHandle`。

### 二、代码示例

#### Task：五次调用不等于五个 Worker

```python
import os
import ray

ray.init()

@ray.remote
def process(x):
    return {"result": x * 2, "pid": os.getpid()}

print(ray.get([process.remote(i) for i in range(5)]))
```

同一个 Worker 可先后执行多个 Task，但任务不保证落在同一个进程，不能依赖全局变量保存跨任务状态。

#### Actor：多次调用共享实例状态

以下示例独立运行：

```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

actor = Counter.remote()
for _ in range(3):
    print(ray.get(actor.increment.remote()))  # 依次输出 1、2、3
```

Actor 重启会重新初始化，内存状态不会自动恢复。Async / Threaded Actor 支持并发调用，但需要处理共享状态同步；并发也不保证 CPU 计算加速。

### 三、Ray Data 中如何选择？

`map_batches()` 传普通函数时使用 Task，传可调用类时使用 Actor 池。无状态计算用函数；需要复用模型或 Tokenizer 时用可调用类。

假设 `ds` 含字符串列 `text`：

```python
from transformers import AutoTokenizer

class TokenizerWorker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def __call__(self, batch):
        return dict(self.tokenizer(batch["text"].tolist()))

result = ds.map_batches(
    TokenizerWorker,
    batch_size=1000,
    batch_format="numpy",
    concurrency=4,
)
```

这里配置 4 个计算 Actor，各初始化一份 Tokenizer，再处理多个 Batch。若把初始化放进普通批处理函数，每次函数调用都会重新构建实例。多个 Actor 的模型内存独立占用，实际并发还受资源和背压限制。

### 四、CPU 实验代码

用重复 SHA-256 模拟 CPU 密集计算，不是真正的分词。完整脚本见 [ray_cpu_test.py](experiments/ray_cpu_test.py)。

```python
import os
import time
import hashlib
import argparse

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
```

连接已有集群运行：

```bash
python experiments/ray_cpu_test.py --address auto --output /tmp/ray_cpu_output
```

计时包含执行阶段的初始化、计算和写出，不含前面的数据生成与 Dataset 构造。集群中的 `/tmp` 属于执行节点；统一访问输出应使用共享存储。

### 五、为什么两个 Actor 会产生 204 条 Task 记录？

可调用类配合 `concurrency=2` 创建两个计算 Actor，反复处理输入；`num_cpus=1` 为每个 Actor 声明 1 个逻辑 CPU，不限制整个作业的 CPU 用量。

![两个 Actor 与多次任务调用](https://cdn.jsdelivr.net/gh/WenDing-Y/ray@5cc415ade7/images/ray-core-actor-task-count.svg)

[单独查看图片](https://github.com/WenDing-Y/ray/blob/work/images/ray-core-actor-task-count.svg)

另一环境的分组截图显示：

| 分组 | 明细 | 数量 |
| --- | --- | ---: |
| Actor 1 | `__init__` 1 + `get_location` 1 + `submit` 49 | 51 |
| Actor 2 | `__init__` 1 + `get_location` 1 + `submit` 51 | 53 |
| Write | 写出任务 | 100 |
| **合计** | **100 次计算 + 100 次写出 + 2 次初始化 + 2 次位置查询** | **204** |

49 / 51 的分配无需严格平分。**Task Table 统计执行记录，不是进程或 Actor 数量。** Batch、Block 和 Task 也不保证一一对应，不能从参数直接推导总数。

### 六、云端实测：为什么是 287？

| 项目 | 结果 |
| --- | --- |
| 环境 | Ray 2.59.0；同一云机器上运行 3 个 Docker Ray 节点，共 4 个任务 CPU |
| Job ID | `02000000` |
| 输入 / 初始 Block / Batch | 100,000 行 / 100 / 1,000 行 |
| 计算 Actor | 2 个，各声明 1 CPU |
| 写出消费阶段 | 57.87 秒，约 1,728 行/秒 |
| MapBatches / Write | 各 100 个任务 |
| 任务快照 / 后续截图 | 287 / 290 条 |

云端同样包含上述 204 条记录，另外多了 **83 条内部辅助记录**：

| 额外记录 | 明细 | 数量 |
| --- | --- | ---: |
| 内部 Actor 创建 | Stats、AutoscalingCoordinator、ActorLocationTracker 各 1 次 | 3 |
| 资源协调 | `get_reserved_resources` 53 + `request_resources` 6 + `cancel_request` 1 | 60 |
| 统计调用 | `update_execution_metrics` 13 + `register_dataset` 1 + `get_datasets` 4 | 18 |
| 位置跟踪 | `ActorLocationTracker.update_actor_location` | 2 |
| **合计** | **204 + 83 = 287** | **83** |

Dashboard 查询仍可能产生统计调用，因此稍后截图为 290 条。两边核心计算数量一致，差异来自额外辅助记录；另一环境为什么未显示这些记录，还需比较版本、配置和统计范围。

![云端 Task Table](https://cdn.jsdelivr.net/gh/WenDing-Y/ray@5cc415ade7/images/ray-core-dashboard-tasks.png)

[单独查看图片](https://github.com/WenDing-Y/ray/blob/work/images/ray-core-dashboard-tasks.png)

Actor 表共 5 个实例：**2 个计算 Actor + 3 个内部辅助 Actor**。`concurrency=2` 只配置本次转换的计算池。

![云端 Actor Table](https://cdn.jsdelivr.net/gh/WenDing-Y/ray@5cc415ade7/images/ray-core-dashboard-actors.png)

[单独查看图片](https://github.com/WenDing-Y/ray/blob/work/images/ray-core-dashboard-actors.png)

计算 Actor 在作业成功后因引用释放而变为 `DEAD`，不代表失败。两个 Actor 在不同容器中恰好同为 PID 2525，应结合 Actor ID、节点或 IP 区分。

核对任务数时，筛选同一个 Job，再按任务名、类型和 Actor ID 分组，对照 `ds.stats()`，检查辅助调用与重试。

实验材料：[运行日志](experiments/ray_cpu_run.log) · [任务快照](experiments/ray_cpu_tasks.json) · [结果汇总](experiments/ray_cpu_summary.json)
