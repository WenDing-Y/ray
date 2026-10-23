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

## 第三题：Ray Data 的数据流水线（Pipeline）如何执行？

Ray Data 把大规模数据组织成多个 **Block**，通过 **Operator** 定义处理阶段，由 **Streaming Executor** 协调，再交给 Ray 的 **Task / Actor** 分布式执行。

核心机制是：**惰性构建计划，消费触发计算，数据块在各阶段之间流水线流动。** 这里的 Pipeline 指处理流水线，不特指旧版的 `DatasetPipeline` API。

### 一、从代码到执行计划

```python
import ray

# clean_text、is_valid 和 TokenizeBatch 需要自行实现。
# TokenizeBatch 是可调用类，其输入应与 batch_format 匹配。
ds = ray.data.read_json("s3://bucket/raw/*.jsonl")
ds = ds.map(clean_text)
ds = ds.filter(is_valid)
ds = ds.map_batches(
    TokenizeBatch,
    batch_size=1000,
    batch_format="numpy",
    concurrency=4,
)
ds.write_parquet("s3://bucket/output/")
```

整个流水线的逻辑如下：

![Ray Data 数据流水线逻辑](https://raw.githubusercontent.com/WenDing-Y/ray/work/images/screenshot-20261023-114345.png)

图中展示的是逻辑处理顺序，并不要求全部 Block 完成一个阶段后才能进入下一阶段。

执行过程可以概括为：

1. **构建逻辑计划**：描述 Read → Map → Filter → MapBatches → Write 等操作。转换通常不会立即处理完整数据集，但数据源和元数据查询可能提前产生工作。
2. **触发执行**：`write_parquet()` 等消费或输出操作启动所需计算。`take()`、迭代 `iter_batches()`、`materialize()` 等也会触发计算；部分查询可能利用元数据完成。
3. **生成并优化物理计划**：把逻辑操作映射为物理算子，按规则进行融合等优化。
4. **调度执行**：Executor 为可执行算子提供输入，算子提交 Task 或 Actor 调用，Ray 运行时安排实际执行位置。
5. **传递结果**：上游产生的 Block 引用交给下游，最终写入目标存储。

### 二、流式执行：不同数据块同时处于不同阶段

不需要等全部数据读取完才开始清洗，也不需要等全部清洗完才开始分词。同一时刻可能出现：

| 数据块 | 当前进度 |
| --- | --- |
| Block 1 | 正在 Tokenize |
| Block 2 | 正在 Filter |
| Block 3 | 正在读取或清洗 |

这是逻辑示意，实际 Map、Filter 等操作可能融合在同一任务内。排序、全局聚合等需要跨分片协调的操作也可能形成阶段性等待。

Streaming Executor 持续收集任务输出、更新算子队列，并结合资源预算和背压决定哪些算子可以继续接收输入。**Streaming 在这里主要指有界数据的流水线执行，不代表 Flink 式无界事件流及其 Watermark、窗口和状态 Checkpoint 语义。**

### 三、Operator、Task、Worker 的分工

| 概念 | 职责 |
| --- | --- |
| Physical Operator | 执行控制单元，组织输入、输出、任务及执行状态 |
| Task / Actor | 实际执行用户函数；Actor 可跨调用保留初始化状态 |
| Worker | 运行 Task 或 Actor 的进程 |
| Block | 被处理和传递的数据分片 |
| Streaming Executor | 协调物理算子的执行进度、资源使用与背压 |

物理算子通常不是独立的 Worker 进程。一个计算算子可以管理多个 Task 或 Actor；一条流水线也不等于一个 Task。

**算子融合（Operator Fusion）** 可以把相邻的 Map、Filter 等兼容操作合并执行，减少任务调度、中间数据物化和传输开销。因此，代码中的三个转换不一定对应三个独立的物理阶段。

### 四、背压：下游慢时限制上游

假设读取速度为 100 Blocks/s，Tokenize 只能处理 10 Blocks/s，中间结果就可能积压。

Ray Data 通过背压限制上游任务提交或数据输出，避免中间数据无限增长。Object Store 内存紧张时，还可能通过 **Spilling** 将对象溢写到磁盘。

背压和 Spilling 不能保证不会 OOM：Worker 堆内存、模型临时对象、GPU 显存和磁盘容量仍然需要单独关注。

### 五、Block 如何从上游传到下游？

以存放在 Object Store 中的普通大型 Block 为例：

1. 上游 Worker 产生数据，Ray 将结果存入所在 **Ray Node 的 Object Store**。
2. Executor 在算子间传递 **ObjectRef 和元数据**，无需先把整个 Block 拉回 Driver。
3. 下游任务需要数据时，Ray 运行时负责定位对象；若本节点没有所需对象，则按需跨节点传输。
4. 数据就绪后，下游 Worker 从本地对象存储读取并处理。

| 场景 | 网络传输 | Worker 读取 |
| --- | --- | --- |
| 同一个 Ray Node 内的不同 Worker | 通常无需网络搬运 | 支持零拷贝的 Arrow / NumPy 缓冲区可共享读取 |
| 不同 Ray Node 首次消费 | 通常需要传输并建立本地副本 | 到达后再从本地读取 |
| 目标节点已有可用副本 | 可复用，不必重复传输 | 本地读取 |

**传递 ObjectRef 不等于复制整个 Block，也不会把数据广播到所有节点。** 零拷贝读取不能消除跨节点网络传输；Python 对象反序列化、格式转换和修改只读数组仍可能分配额外内存。小对象内联、溢写恢复等路径与上述简化过程有所不同。

在 Kubernetes 中，不同 Ray Pod 通常属于不同 Ray Node，各自拥有独立的 Object Store。即使两个 Pod 在同一台物理机器，也不会自动共享同一个对象存储。

### 六、性能优化关注点

- **减少中间数据**：业务语义允许时先过滤无效记录、移除无用字段，再进行昂贵计算。
- **利用算子融合和数据本地性**：减少中间物化及跨节点传输，同时兼顾资源利用率；具体放置受调度策略和 Actor 所在节点影响。
- **调整 Block、Batch 和并发**：过大可能增加内存压力，过小可能放大调度开销；三者不是同一个参数。
- **用指标验证**：通过执行后的 `ds.stats()`、Ray Dashboard 和对象存储指标定位瓶颈。

记住这条链路即可：**Dataset 描述计划 → Executor 协调 Operator → Task / Actor 在 Worker 中处理 Block → Object Store 保存中间对象 → 下游按需获取并继续处理。**

## 第四题：Ray Data 中的数据倾斜常见吗？

比较常见，尤其是在大规模文本清洗、Tokenize 和模型推理中。但具体是否发生、哪种倾斜更突出，取决于数据分布和处理逻辑，不能仅凭数据规模判断。

**数据量均衡，不代表计算耗时均衡。** 对 AI 数据流水线而言，文本长度、图片尺寸、模型生成长度等差异，都可能导致少数任务特别慢。

### 一、常见的倾斜类型

| 类型 | 典型原因 | 可能表现 |
| --- | --- | --- |
| 数据分片不均 | 输入文件大小不同、读取分块不均、转换后数据膨胀 | 少数 Block 或任务输入特别大 |
| 计算耗时不均 | 文本长短、样本复杂度、推理输出长度不同 | 输入行数相近，但 Task 耗时差异很大 |
| Shuffle / Key 倾斜 | 聚合时存在热 Key，排序分区不均 | 少数交换或聚合任务处理大量数据 |
| 执行资源不均 | 节点性能、资源竞争、Actor 放置或本地性约束 | 部分 Worker 忙，部分 Worker 空闲 |

资源负载不均也可能来自调度或运行环境，并不一定是数据本身倾斜。`repartition()` 也不等于一定按 Key 分区，需要区分其具体模式和执行策略。

例如两个 Block 都有 1,000 条文本：

- Block A：每条约 100 个字符。
- Block B：每条约 10,000 个字符。

Tokenize 时，Block B 可能远慢于 Block A。因此，行数、字节数和实际计算成本应分别观察。

### 二、动态调度为什么不能消除倾斜？

![Ray Data 均衡任务与长尾任务对比](https://raw.githubusercontent.com/WenDing-Y/ray/work/images/ray-data-task-skew.svg)

对于可独立调度的 Task，Worker 空闲后可以继续执行其他任务，因此动态调度能够缓解任务数量或耗时不均。但它不能自动拆开一个已经运行的慢 Task，让其他 Worker 接手其中一部分。

例如有 100 个任务，其中 99 个耗时 2 秒，最后一个耗时 100 秒。即使其他任务已经完成、集群还有空闲资源，作业仍然需要等待最后这个任务。

这就是 **长尾任务**。对于有状态 Actor，任务还会受到 Actor 实例所在节点、可用实例及其执行能力等限制，不能完全按无状态 Task 理解。

### 三、与 Flink 的区别

| 对比 | Ray Data | Flink |
| --- | --- | --- |
| 典型场景 | 有界数据、AI 数据处理 | 持续流处理，也支持批处理 |
| 常见倾斜来源 | Block 大小、样本计算成本、聚合 Key 分布 | 热 Key、分区负载、单条记录处理成本 |
| 普通转换的分配方式 | 通过 Task / Actor 处理分片 | 通常由持续运行的 Subtask 消费分区数据 |
| 主要影响 | 长尾、吞吐量下降、资源利用不均，也可能产生背压 | 延迟升高、背压、状态增长 |
| 常见缓解方向 | 调整任务粒度、按成本分批、处理异常样本 | Key 打散、预聚合、调整分区与并行度 |

普通 `map_batches()` 不要求按 Key 固定路由，因此不必一开始就寻找热 Key；但执行 `groupby()` 聚合时，同样可能受到热 Key 影响。两者都可能遇到计算成本不均，区别不是绝对的。

### 四、如何排查和缓解？

针对 JSONL → 清洗 → Tokenize → Parquet，建议按以下顺序检查：

1. **看输入分片**：比较文件、Block 和任务输入的行数、字节数。少数分片过大时，调整读取和分块策略，必要时重分区；也要考虑额外数据搬运成本。
2. **看计算成本**：统计文本长度等特征，检查慢任务是否集中处理超长或异常样本。业务允许时截断，或将长样本分流处理。
3. **看批次构造**：调整 `batch_size`，控制一次调用的内存和计算量。模型推理可考虑按长度分桶或限制每批总 Token 数；固定行数不代表成本相同。
4. **看资源与并发**：检查 CPU/GPU 利用率、Actor 分布、对象传输、Spilling 和背压。增加并发只在存在可并行工作且资源充足时有效，不能直接加速单个不可拆分的慢任务。
5. **有聚合时再看 Key 分布**：检查热 Key；若聚合语义允许，可采用局部预聚合，或加盐后分两阶段聚合。

使用执行后的 `ds.stats()`、Ray Dashboard 和任务指标观察耗时分布，关注最长任务与典型任务的差距。任务耗时还可能包含数据获取和资源等待，不能直接把所有慢任务都归因于用户函数。

**减小 Batch 不等于增加 Task 数量，也不保证消除长尾。** 一个 Task 可能连续处理多个 Batch；只有批次或任务划分真正改变了工作分配，才可能改善负载均衡。

对于大规模 LLM 数据预处理，优先沿着 **数据量是否均衡 → 计算耗时是否均衡 → 资源利用是否均衡** 排查；文本长度造成的计算倾斜，通常值得重点关注。

## 第五题：Ray Data Pipeline 执行缓慢，如何系统排查？

**先定位限制吞吐量的 Operator，再判断计算、数据、调度、资源或 I/O 原因，最后针对瓶颈优化并重新测量。** 不要一开始就增加 Worker 或修改所有参数。

### 一、排查全流程

![Ray Data Pipeline 性能排查流程](https://raw.githubusercontent.com/WenDing-Y/ray/work/images/ray-data-performance-triage.svg)

对于 1TB JSONL → 清洗 → Tokenize → Parquet，优先关注 Task 耗时、Block 分布、CPU/GPU 利用率、Object Store、Spilling 和背压。

### 二、定位哪个 Operator 限制整体吞吐量

```python
import ray

# clean_text、is_valid、Tokenize 需要自行实现。
# Tokenize 是可调用类，接收 NumPy 格式批次。
ds = ray.data.read_json("s3://bucket/input/")
ds = ds.map(clean_text)
ds = ds.filter(is_valid)
ds = ds.map_batches(
    Tokenize,
    batch_size=1000,
    batch_format="numpy",
    concurrency=4,
)
ds.write_parquet("s3://bucket/output/")
print(ds.stats())
```

| 逻辑阶段 | 优先检查 |
| --- | --- |
| ReadJSON | S3 吞吐、文件数量与大小、读取并行度、JSON 解析开销 |
| Map(clean_text) | Python 逐行开销、正则等 UDF 计算 |
| Filter(is_valid) | 判断成本、过滤比例、下游实际输入量 |
| MapBatches(Tokenize) | Tokenizer 效率、Batch、Actor 并发、线程竞争 |
| WriteParquet | 编码与压缩、S3 写入、输出文件数量 |

实际物理算子可能融合，不一定与表中的逻辑步骤一一对应。流式执行时，各阶段可以重叠；**算子累计 Task 耗时不能直接相加作为作业总耗时**，累计值甚至可能大于作业墙钟时间。

![流水线吞吐瓶颈示例](https://raw.githubusercontent.com/WenDing-Y/ray/work/images/ray-data-throughput-bottleneck.svg)

比较处理能力时，需要统一口径。过滤后的输出条数较少、Tokenize 后数据体积变大，都可能使直接比较各阶段“条/秒”产生误判。

### 三、监控工具与关键指标

#### 1. ds.stats() 与日志

在消费或输出操作执行后检查 `ds.stats()`。需要更详细的统计时，在执行前设置：

```python
from ray.data import DataContext

DataContext.get_current().verbose_stats_logs = True
```

结合执行进度条、Ray 会话日志中的 `ray-data.log` 和 Worker 日志观察。具体指标名称、可见范围及采集方式随版本和配置变化，写出操作的完整信息也应结合 Dashboard 和日志检查。

| 指标或观察项 | 用途 |
| --- | --- |
| Remote wall time | 比较任务执行墙钟耗时及长尾 |
| Remote CPU time | 判断任务消耗的 CPU 时间；多线程情况下可能超过墙钟时间 |
| UDF time | 判断用户函数是否主导计算成本 |
| Output block size / rows | 检查输出分片大小、行数与数据膨胀 |
| Peak heap memory | 观察 Worker 堆内存压力，不等于 Object Store 或 GPU 显存 |
| Tasks per node | 检查任务数量分布，需结合节点性能和任务耗时解释 |
| 队列、Spilling、吞吐量 | 检查上下游失衡、对象存储压力和 I/O 成本 |

例如任务耗时 `min=1s、max=120s、mean=5s`，说明存在明显长尾，但仍要检查是异常样本、数据倾斜、慢节点还是 I/O。

若相同口径下墙钟耗时远高于 CPU 时间，可优先检查 I/O、阻塞或设备等待；不能仅凭这一点就断定是调度或背压。任务启动前的 Pending 时间，应单独从任务状态和时间线观察；GPU 工作也不会直接体现为 CPU 消耗。

#### 2. Ray Dashboard

Kubernetes 环境中，可按实际命名空间和 Head Service 名称执行：

```bash
kubectl -n <namespace> port-forward svc/<ray-head-service> 8265:8265
```

在浏览器打开 `http://localhost:8265`，结合当前版本可用的 Jobs / Ray Data、Cluster、Tasks 和 Metrics 页面观察：

- 算子进度、输入输出队列与吞吐量。
- 实际 CPU/GPU 利用率、内存和节点分布；Ray 声明或分配的逻辑资源不等于实际使用率。
- Task 的 Running、Pending、失败、重试和长尾情况。
- Object Store、Spilling、磁盘及网络 I/O。

进度条中的 `[backpressure]` 表示算子受到下游消费或缓冲限制，**不一定是该算子本身慢**。应顺着流水线检查下游和资源预算。

### 四、按症状缩小范围

| 症状 | 优先检查 | 可能的优化方向 |
| --- | --- | --- |
| CPU 低 | 输入是否充足、并发是否受限、I/O 或设备等待 | 改善供给、读取并行度或批处理，确认瓶颈后调整并发 |
| CPU 高 | UDF 计算、线程竞争、序列化 | 向量化、批量调用、减少重复工作、控制库线程数 |
| 内存持续增长 | Worker 堆、Object Store、GPU 显存分别增长多少 | 控制工作集、Batch、Block 和并发，排查缓存或泄漏 |
| 少数 Task 特别慢 | 样本长度、输入字节数、节点和 I/O 差异 | 分流异常样本、调整任务粒度、检查慢节点 |
| 大量 Pending | CPU/GPU/自定义资源请求、节点标签、Actor 放置及集群容量 | 修正不可满足的资源请求或放置约束，必要时扩容 |
| Spilling 严重 | 中间对象工作集、下游速度、对象引用保留、磁盘性能 | 减少中间数据、控制并发与分块、改善内存或磁盘配置 |

Pending 不一定只因 CPU 不够，资源空闲也可能无法满足某项约束；Spilling 也不一定都是错误，但大量溢写与恢复可能显著拖慢执行。

### 五、优化顺序与验证

1. **建立基线**：记录总耗时、输入量、有效输出量、吞吐量、资源峰值与成本。
2. **定位瓶颈**：结合算子统计、任务时间线和日志，确认限制整体吞吐量的阶段。
3. **检查工作分配**：观察任务粒度、Block/Batch、Pending、并发和数据倾斜。
4. **优化 UDF 与初始化**：批量 Tokenize、向量化；需要复用昂贵初始化时使用可调用类 Actor，避免每批加载模型。
5. **检查存储与网络**：关注 S3、输出文件数量、对象传输、Spilling 和磁盘吞吐。
6. **控制变量复测**：一次只改少量变量，在同一数据集和可比资源下重复测量，并区分冷启动与稳定运行。

增加 Worker 不一定能提高吞吐量：S3、下游写入、固定 Actor 数量或不可拆分的长尾任务，都可能让新增 CPU 闲置。

## 第六题：Ray Object Store 是什么？

**Object Store 是每个 Ray Node 上用于存放和共享对象的存储系统。** 同节点 Worker 可共享数据，跨节点由 Ray 运行时按需传输对象。Ray 的大型共享对象通常使用基于 Plasma 的共享内存存储。

对于 Ray Data，可以理解为：**上游 Task 产生 Block → Object Store 保存 → 下游 Task 按引用获取并处理**。相邻算子可能融合，并不保证每一步都物化出独立 Block。

### 一、同节点共享，跨节点传输

![Ray Object Store 同节点共享与跨节点传输](images/ray-data-object-store.svg)

- 每个 Ray Node 拥有自己的 Object Store，不同节点不直接共享同一块物理内存。
- 同节点 Worker 读取支持零拷贝的 NumPy / Arrow 缓冲区时，可以共享底层数据。
- 跨节点消费通常需要网络传输，目标节点保存对象副本，Worker 再从本地读取。已有可用副本时可复用。
- `ObjectRef` 是引用，不是对象数据；传递引用不等于复制对象或广播数据。

图中是普通大型共享对象的典型路径；小对象内联、溢写恢复等情况有所不同。

### 二、示例：两个 Task 读取同一个数组

```python
import numpy as np
import ray

ray.init()

# 80,000,000 字节，约 80 MB（76.3 MiB）。
data = np.ones(10_000_000, dtype=np.float64)
ref = ray.put(data)

@ray.remote
def process(arr):
    return arr.sum()

a = process.remote(ref)
b = process.remote(ref)
print(ray.get([a, b]))  # [10000000.0, 10000000.0]
```

`ray.put()` 将数组放入 Object Store，返回引用。作为顶层任务参数传入 `ref` 时，Ray 会解析依赖，`process()` 接收到的是数组。

如果两个 Task 在对象所在节点执行，可以共享读取同一份数组，不必为每个 Worker 复制一份。这个示例没有固定任务位置，实际是否发生跨节点传输取决于调度。

注意两个边界：

- **零拷贝不适用于所有对象。** 普通 Python 对象通常需要反序列化，格式转换也可能分配新内存。
- **共享对象不可原地修改。** 从 Object Store 读取的 NumPy 数组通常只读；需要修改时先 `arr.copy()`，得到 Worker 内的可写副本，再按需输出新对象。最初 `ray.put()` 前的本地数组不等于共享对象本身。

### 三、Ray Data 中的使用

```python
# transform 需要自行实现。
ds = ray.data.read_parquet("s3://bucket/input/")
ds = ds.map_batches(transform)
ds.write_parquet("s3://bucket/output/")
```

未融合时，可用下面的概念链路理解：

```text
Read Task → Block A → MapBatches Task → Block B → Write Task
               ↑                          ↑
         Object Store 中的输入      Object Store 中的输出
```

产生新结果时，Block A 和 Block B 是不同的逻辑对象，但部分底层缓冲区可能共享；不能据此断言每次转换都会完整复制数据。算子融合还可能省去中间结果的独立物化。

### 四、Object Store 与 Worker Heap

| 对比 | Object Store | Worker Heap |
| --- | --- | --- |
| 主要内容 | 共享 Ray 对象、Ray Data Block | Python 对象、模型、临时计算结果 |
| 共享范围 | 同节点多个 Worker 可读取同一对象 | 通常属于单个进程 |
| 零拷贝 | 支持部分数据类型的共享读取 | 转换或计算仍可能产生额外分配 |
| 内存不足 | 符合条件的对象可 Spilling 到磁盘 | 可能导致 Worker OOM |
| 排查重点 | 中间工作集、引用保留、溢写与恢复 | Batch、模型大小、临时对象及并发 |

**Spilling 只缓解对象存储压力，不能代替 Worker 堆或 GPU 显存管理。** 大量溢写和恢复会增加磁盘 I/O；对象被使用或固定时也不保证随时能释放，仍可能遇到内存或磁盘不足。

### 五、KubeRay 中的边界

如果 Head Pod 和两个 Worker Pod 各运行一个 Ray Node，它们通常各有独立的 Object Store。

即使两个 Pod 位于同一台 Kubernetes 物理节点，也不会自动合并对象存储。Pod 内的 Worker 可共享本节点对象；跨 Ray Node 获取对象仍需要对象传输，不能视为同节点共享内存读取。
