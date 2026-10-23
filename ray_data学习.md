# Ray Data

## 第一题：解释 Ray Data 核心概念

### 一、Ray Data 核心概念总览

| 层次 | 核心概念 | 作用 |
| --- | --- | --- |
| 数据模型 | Dataset | 用户操作的分布式数据集 |
| 数据模型 | Block | 实际存储和处理的数据分块 |
| 数据模型 | Block Metadata | 记录行数、大小、Schema 等信息 |
| 执行模型 | Logical Plan | 描述要执行哪些操作 |
| 执行模型 | Physical Plan | 将逻辑操作转换成物理算子后的执行计划 |
| 执行模型 | Operator | 组织具体的数据处理过程 |
| 执行模型 | Streaming Executor | 流水线式调度物理算子 |
| 算子模型 | map / filter | 逐行处理 |
| 算子模型 | map_batches | 批量处理 |
| 算子模型 | groupby / sort / repartition | 聚合、排序与数据重分布 |
| 调度模型 | Task | 一次具体的计算任务 |
| 调度模型 | Actor | 可复用的有状态计算进程 |
| 调度模型 | Worker | 执行 Task 或 Actor 的进程 |
| 内存模型 | Object Store | 分布在各节点上的对象存储 |
| 内存模型 | ObjectRef | 对象引用 |
| 内存模型 | Spilling | 对象存储内存不足时将对象溢写到磁盘 |
| 内存模型 | Backpressure | 控制上下游处理速度 |
| 容错模型 | Lineage | 记录数据的计算依赖关系 |
| 容错模型 | Retry | 失败后重试任务或计算 |

其中最重要的 8 个概念是：

**Dataset、Block、Task、Operator、Streaming Executor、Object Store、Actor、Backpressure。**

理解它们，基本就能建立 Ray Data 的整体架构认知。

### 二、通过一个例子理解整个执行过程

假设使用 Ray Data 处理 1TB JSONL 数据：

```python
import ray

# clean_text、is_valid、tokenize_batch 为需要自行实现的处理函数。
ds = ray.data.read_json("s3://bucket/input/")
ds = ds.map(clean_text)
ds = ds.filter(is_valid)
ds = ds.map_batches(
    tokenize_batch,
    batch_size=1000,
    concurrency=4,
)
ds.write_parquet("s3://bucket/output/")
```

简化后的执行架构：

```text
                         Driver / Dataset
              构建 Logical Plan → 优化 → Physical Plan
                                 │
                                 ▼
                       Streaming Executor
               协调算子、任务调度、背压和资源使用
                                 │
                                 ▼
       Read       →     Map / Filter   →   Tokenize   →   Write
     Operator 1          Operator 2        Operator 3    Operator 4
                                 │
                    由 Ray 调度到各节点执行
                                 ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │   Ray Node A    │ │   Ray Node B    │ │   Ray Node C    │
     │  Worker / Task  │ │  Worker / Task  │ │  Worker / Task  │
     │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
     │ │ Object Store│ │ │ │ Object Store│ │ │ │ Object Store│ │
     │ │    Blocks   │ │ │ │    Blocks   │ │ │ │    Blocks   │ │
     │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
     └─────────────────┘ └─────────────────┘ └─────────────────┘
```

概念示意：各算子可在不同节点并发执行；实际物理算子可能经过融合优化。

这里需要特别区分：

- **Block 是数据单位**，一个 Block 包含多行数据。
- **Task 是执行单位**，一次 Task 可以处理一个或多个 Block，也可能产生多个 Block。
- **Operator 是计算阶段**，负责组织具体计算；执行计算的物理算子可以管理一组 Task 或 Actor。
- **Executor 是调度协调者**，决定什么时候给 Operator 分配输入和启动计算；具体任务的节点放置由 Ray 运行时负责。

Ray Data 采用流水线执行，不需要等待所有数据完成 Read 才启动后面的 Map。

### 三、容易混淆的几个概念

| 概念 | 容易产生的误解 | 正确认识 |
| --- | --- | --- |
| Dataset | 所有数据都在 Driver 内存里 | Driver 主要持有数据集描述和引用 |
| Block | 一个 Block 对应一个 Task | 不保证一一对应 |
| Batch | 一个 Batch 就是一个 Block | Batch 是用户函数处理的数据批次 |
| Task | 一个 Task 就是一个 Worker | Worker 进程可以先后执行多个 Task |
| Actor | 和 Task 没区别 | Actor 能跨多次调用保留状态 |
| ObjectRef | 就是实际的数据 | 它是定位远程对象的引用 |
| Streaming | 等同 Flink 无界流处理 | 主要是有界数据的流水线执行 |
| Spilling | 内存不足就一定不会 OOM | Object Store 可以溢写，但 Worker 堆内存仍可能 OOM |

。
