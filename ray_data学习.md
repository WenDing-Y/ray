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

![Ray Data 简化执行架构](https://raw.githubusercontent.com/WenDing-Y/ray/work/images/screenshot-20261023-111725.png)

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

## 第二题：map() 和 map_batches() 有什么区别？

Ray Data 中批处理接口的名称是 `map_batches()`，不是 `batch_map`。最核心的区别是用户函数的调用粒度：

- `map()`：每次处理一行数据。
- `map_batches()`：每次处理一批数据，可以利用 NumPy、PyArrow、Pandas 等进行批量计算。

**`map_batches()` 不一定比 `map()` 快。** 如果内部仍然逐行循环，性能优势可能很小。批量接口能否减少底层操作次数、是否利用向量化，以及数据格式转换开销，都会影响实际性能。

### 一、核心区别

| 对比 | map() | map_batches() |
| --- | --- | --- |
| 处理粒度 | 一行 | 一批 |
| 用户函数输入 | 表示一行的 dict | NumPy 字典、Pandas DataFrame、PyArrow Table 等 |
| 每次函数调用 | 1 行 | 指定 batch_size 时，通常为该行数；尾批次可能较小 |
| 数据处理方式 | 逐行 | 可批量计算，也可在函数内逐行处理 |
| Python 用户函数调用次数 | 通常每行一次 | 通常每批一次 |
| 向量化计算 | 不方便 | 非常适合 |
| 模型批量推理 | 难以直接合并多行输入 | 非常适合 |
| 典型场景 | 字段清洗、简单转换 | Tokenize、特征计算、模型推理 |

`batch_size` 是输入批次大小，不限制函数输出行数；`map_batches()` 可以改变行数。未指定 `batch_size` 时，也不要假设每次调用具有固定行数。

### 二、完整例子

假设有 10,000 条记录，每条包含 `user_id` 和 `amount`，需要计算 `score = amount * 2`。

#### 1. 使用 map()

```python
import ray

ray.init()

ds = ray.data.from_items([
    {"user_id": i, "amount": i}
    for i in range(10000)
])

def process_row(row):
    return {**row, "score": row["amount"] * 2}

result = ds.map(process_row)
print(result.take(3))
```

输出：

```python
[
    {"user_id": 0, "amount": 0, "score": 0},
    {"user_id": 1, "amount": 1, "score": 2},
    {"user_id": 2, "amount": 2, "score": 4},
]
```

如果完整消费全部 10,000 行，在没有重试或重复执行的情况下，`process_row()` 会调用 10,000 次。这里的 `take(3)` 只请求前三行，不保证会处理整个数据集；底层按块执行，也可能处理多于三行。

#### 2. 使用 map_batches()

以下代码可作为另一个独立脚本运行：

```python
import ray

ray.init()

ds = ray.data.from_items([
    {"user_id": i, "amount": i}
    for i in range(10000)
])

def process_batch(batch):
    return {**batch, "score": batch["amount"] * 2}

result = ds.map_batches(
    process_batch,
    batch_size=1000,
    batch_format="numpy",
)
print(result.take(3))
```

输出相同。这里 NumPy 字典中的每个值是一列数组，`batch["amount"] * 2` 会对整列执行向量化计算。返回新字典和新计算结果，也避免原地修改可能只读的输入数组。

完整消费时，如果恰好组成 10 个完整批次，用户函数只需调用 10 次。实际调用次数还取决于数据分块、任务输入组合和尾批次，不能仅用总行数除以 `batch_size` 确定。

### 三、内部执行方式有什么区别？

假设一个 Block 正好包含 1,000 行，且 `batch_size=1000`，可以这样理解用户函数调用过程：

```text
map()
  Block：1000 行
    process_row(第 1 行)
    process_row(第 2 行)
    ...
    process_row(第 1000 行)
  用户函数共调用 1000 次

map_batches()
  Block：1000 行
    process_batch(包含 1000 行的批次)
  此例中用户函数调用 1 次，内部进行 NumPy 批量计算
```

**`map()` 并不是每行启动一个 Ray Task。** Ray 可能使用一个 Task 处理一个 Block，在 Task 内调用 `process_row()` 1,000 次；`map_batches()` 也可能使用一个 Task，只是在 Task 内调用一次批处理函数。

因此，两者的主要区别是用户函数的调用粒度，不能直接据此推导 Ray Task 的数量。实际任务划分还受 Block 组合、算子融合和执行策略影响。

### 四、如果 map_batches() 内部仍然逐行循环呢？

```python
import numpy as np

def process_batch_loop(batch):
    scores = []
    for amount in batch["amount"]:
        scores.append(amount * 2)
    return {**batch, "score": np.array(scores)}
```

虽然使用批处理接口，这个函数仍然执行大量 Python 层循环。对比向量化版本：

```python
def process_batch_vectorized(batch):
    return {**batch, "score": batch["amount"] * 2}
```

第二种把数组运算交给 NumPy 底层实现，通常更快。但实际收益应通过具有代表性的数据和完整消费过程测量，不能只比较调用次数。

### 五、函数调用次数估算

假设各 Block 的行数尽量均匀，且每个 Block 单独切成 Batch，Batch 不跨 Block 边界：

| 参数 | 数值 |
| --- | --- |
| 总行数 | 10,000 |
| batch_size | 1,000 |
| Block 数量 | 8 |
| 每个 Block 行数 | 1,250 |
| map() 用户函数调用次数 | 10,000 |
| map_batches() 用户函数调用次数估算 | 8 × 向上取整(1250 / 1000) = 16 |
| 忽略 Block 边界时的最少批次数 | 向上取整(10000 / 1000) = 10 |

这只是给定假设下的估算，不是 Ray Data 的固定调度规则。实际执行可能组合多个 Block，任务边界也会影响批次构造；失败重试和重复消费还可能增加调用次数。**调用次数不等于运行时间。**

### 六、什么时候应该选择哪个？

选择 `map()`：简单字段转换、规则清洗和逐行处理，尤其是函数无法向量化时。只需要保留或丢弃记录的逐行判断，通常使用 `filter()`。

选择 `map_batches()`：批量 Tokenize、NumPy/Pandas 向量化计算、GPU 推理和批量数据库查询。需要复用模型或 Tokenizer 等昂贵初始化资源时，可以传入可调用类，使用 Actor 保留状态；状态复用不是仅靠增大 Batch 实现的。

结合第一题的 1TB JSONL → 清洗 → Tokenize → Parquet 流水线：

```python
# clean_text、is_valid 和 TokenizerWorker 需要自行定义。
# TokenizerWorker 是实现 __call__ 的类，可在 __init__ 中加载 Tokenizer。
ds = ray.data.read_json("s3://bucket/input/")

# 简单清洗：逐行处理。
ds = ds.map(clean_text)

# 过滤：逐行判断。
ds = ds.filter(is_valid)

# Tokenizer：Actor 复用初始化状态，并批量处理输入。
ds = ds.map_batches(
    TokenizerWorker,
    batch_size=1000,
    concurrency=4,
)

ds.write_parquet("s3://bucket/output/")
```

`TokenizerWorker.__call__` 的输入格式需要与 `batch_format` 匹配，使用具体库时应显式选择合适的格式。

值得记住：`map_batches()` 不只是减少 Python 函数调用次数。对于 Tokenizer 或模型推理，它还可能把多个输入合并成一次底层批量运算，这通常才是显著提升吞吐量的关键。
