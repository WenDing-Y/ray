# Ray Data 官方用户指南：带中文注释的操作示例

来源：[Ray Data User Guides](https://docs.ray.io/en/latest/data/user-guide.html)。整理日期：2026-10-28。

本文件按指南的数据操作分类编写原创小例子；不是全量 API Reference 的逐参数复刻。每段代码带中文注释。
优先覆盖创建、读取、检查、转换、聚合、迭代、连接、洗牌、混合、写出，再覆盖图像、文本、张量、训练和 LLM 等专题，以及执行配置与自定义文件。

## 使用方法

- 先运行“公共准备代码”，后续基础例子默认使用其中的 `ds`、`ROOT`、`pdf` 等变量。每段从公共输入开始，除明确说明外无需串联前一段。
- 标为“本地”的例子使用小数据；其他例子标题旁注明依赖或外部环境。含占位对象、存储路径或环境变量的模板，必须替换并准备环境后才能执行。
- 公开 `latest` 页面会更新，同为 Ray 2.59.0 的本地源码与已发布 wheel 也可能有功能差异，例如 `from_pydict`、`read_hive`。遇到缺失接口先核对安装版本；不要用私有 API 强行替代。
- `take`、迭代或写出触发所需计算。仅构建转换不证明执行过；`count()` 可能走元数据优化。
- 本地 `/tmp` 类路径适用于单机练习；多节点需共享文件系统或对象存储。训练、外部服务和 GPU 例子不默认运行，避免误触服务或占用资源。

基础安装示例（在虚拟环境中执行，版本按项目锁定）：

```bash
python -m pip install "ray[data]" pandas pyarrow polars
# 图像额外安装 pillow；训练额外安装 torch 或 tensorflow。
# LLM 依赖需按 Ray / vLLM / CUDA 兼容矩阵安装，不要盲目混装最新版。
```

## 目录

1. [创建与读取数据](#section-1)
2. [检查数据与执行结果](#section-2)
3. [逐行、批次和字段转换](#section-3)
4. [聚合统计](#section-4)
5. [迭代、切分与消费](#section-5)
6. [Join：数据连接](#section-6)
7. [Shuffle 与重分区](#section-7)
8. [加权数据集混合](#section-8)
9. [写出与转换到其他库](#section-9)
10. [图像数据操作](#section-10)
11. [文本数据操作](#section-11)
12. [张量与 NumPy 操作](#section-12)
13. [PyTorch 训练与批量推理](#section-13)
14. [LLM Processor 数据操作](#section-14)
15. [内存、监控与执行配置](#section-15)
16. [高级转换与自定义文件格式](#section-16)
17. [并发数据集与端到端流水线](#section-17)
18. [补充：组合、表达式与多模态操作](#section-18)

## 公共准备代码

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import ray

# 单机练习启动两个逻辑 CPU；连接已有集群时改为 ray.init(address="auto")。
ray.init(num_cpus=2, include_dashboard=False)
ROOT = Path(tempfile.mkdtemp(prefix="ray-data-ops-"))
rows = [
    {"id": i, "group": "A" if i % 2 == 0 else "B",
     "score": float(40 + i * 10), "text": f" ray data {i} "}
    for i in range(6)
]
pdf = pd.DataFrame(rows)
# 后续示例各自从 ds 开始；转换一般返回新 Dataset，不修改 ds。
ds = ray.data.from_items(rows, override_num_blocks=2)
# 准备本地输入，不依赖公网示例数据。
pdf.to_parquet(ROOT / "input.parquet", index=False)
pdf.to_csv(ROOT / "input.csv", index=False)
(ROOT / "input.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
)
(ROOT / "input.txt").write_text("Ray Data\nHello Ray\n", encoding="utf-8")
np.save(ROOT / "input.npy", np.arange(12).reshape(6, 2))
```

基础输入为 6 行，`id=0..5`，`score=40..90`，`group` 交替为 A/B。结果没有显式排序时，不要依赖分布式任务的输出顺序。

<a id="section-1"></a>
## 1. 创建与读取数据

官方章节：[查看原文](https://docs.ray.io/en/latest/data/loading-data.html)。

### 001. from_items：Python 记录〔本地〕

输入是一组记录，每个字典对应一行。

```python
# 明确指定列名，比传入裸标量更容易理解 Schema。
created = ray.data.from_items([{"x": 1}, {"x": 2}])
assert created.take_all() == [{"x": 1}, {"x": 2}]
```

### 002. range：生成整数〔本地〕

id 从 0 开始。

```python
# 生成 0、1、2、3，不需要先在 Driver 创建列表。
print(ray.data.range(4).take_all())
```

### 003. range_tensor：生成张量〔本地〕

每行是指定形状的张量。

```python
# 生成 3 个形状为 (2, 2) 的张量；查看一批数据。
print(ray.data.range_tensor(3, shape=(2, 2)).take_batch(2))
```

### 004. from_pandas：导入 DataFrame〔本地〕

大数据不要先集中装进 Driver。

```python
# pdf 来自公共准备代码。
print(ray.data.from_pandas(pdf).take(2))
```

### 005. from_arrow：导入 Arrow Table〔本地〕

适合列式数据交换。

```python
# 从已有 DataFrame 转 Arrow，再交给 Ray。
table = pa.Table.from_pandas(pdf, preserve_index=False)
print(ray.data.from_arrow(table).take(2))
```

### 006. from_numpy：导入 NumPy〔本地〕

数组的第一维作为记录维度。

```python
# 6 行，每行向量长度为 2；默认列名为 data。
print(ray.data.from_numpy(np.arange(12).reshape(6, 2)).take_batch(2))
```

### 007. from_pydict：导入列字典〔较新接口〕

各列长度必须一致。

```python
# 两列、两行。
print(ray.data.from_pydict({"x": [1, 2], "y": [3, 4]}).take_all())
```

### 008. read_parquet：读取 PARQUET〔本地〕

本地练习文件已由公共准备代码生成。

```python
# read_* 返回 Dataset；take() 消费少量结果。
loaded = ray.data.read_parquet(str(ROOT / "input.parquet"))
print(loaded.take(2))
```

### 009. read_csv：读取 CSV〔本地〕

本地练习文件已由公共准备代码生成。

```python
# read_* 返回 Dataset；take() 消费少量结果。
loaded = ray.data.read_csv(str(ROOT / "input.csv"))
print(loaded.take(2))
```

### 010. read_json：读取 JSON〔本地〕

本地练习文件已由公共准备代码生成。

```python
# read_* 返回 Dataset；take() 消费少量结果。
loaded = ray.data.read_json(str(ROOT / "input.jsonl"))
print(loaded.take(2))
```

### 011. read_text：按行读取文本〔本地〕

返回 text 列；不是自动按 JSON 解析。

```python
# 每个非空文本行是一条记录。
print(ray.data.read_text(str(ROOT / "input.txt")).take_all())
```

### 012. read_binary_files：读取原始字节〔本地〕

适合自定义文件解析。

```python
# bytes 列保存文件内容；这里的小文本文件只是演示。
print(ray.data.read_binary_files(str(ROOT / "input.txt")).take(1))
```

### 013. read_numpy：读取 .npy〔本地〕

由数组恢复张量记录。

```python
# input.npy 由公共准备代码生成。
print(ray.data.read_numpy(str(ROOT / "input.npy")).take_batch(2))
```

### 014. S3 / HDFS 与读取参数〔外部存储〕

路径必须对执行读取的节点可访问；需要对应凭据与文件系统依赖。

```python
# 投影减少读取列；不要为了提高并行度无限增加 Block。
cloud = ray.data.read_parquet(
    "s3://bucket/input/", columns=["id", "text"], override_num_blocks=8
)
print(cloud.take(1))
# HDFS 同样可通过支持的文件系统读取；需正确配置 HDFS 客户端。
# hdfs = ray.data.read_parquet("hdfs://namenode:8020/data/input/")
```

### 015. 压缩文件与文件列表〔本地〕

为实际存在的压缩文件替换路径；扩展名识别取决于格式。

```python
# gzip CSV 由 pandas 生成，随后读取。
pdf.to_csv(ROOT / "compressed.csv.gz", index=False, compression="gzip")
print(ray.data.read_csv([str(ROOT / "compressed.csv.gz")]).take(1))
```

### 016. read_images / read_tfrecords / read_zarr〔格式依赖〕

这三种文件各给一个入口示例；需要 Pillow、TensorFlow 或 Zarr 等格式依赖和输入文件。

```python
# 图像记录通常包含 image 数组；不同尺寸可能形成变长张量。
images = ray.data.read_images("/shared/images/", mode="RGB")
print(images.take(1))
# TFRecord 需要与数据相匹配的解析约定。
records = ray.data.read_tfrecords("/shared/records/")
print(records.take(1))
# Zarr 路径须在各执行节点可访问。
zarr_data = ray.data.read_zarr("/shared/tensors.zarr")
print(zarr_data.take(1))
```

### 017. from_huggingface：导入 HF Dataset〔datasets〕

需要 datasets；不在示例中下载外部数据集。

```python
from datasets import Dataset as HFDataset
# 先创建很小的 Hugging Face Dataset，再交给 Ray。
hf = HFDataset.from_dict({"text": ["Ray", "Data"]})
print(ray.data.from_huggingface(hf).take_all())
```

### 018. from_torch / from_tf：导入训练数据〔PyTorch / TensorFlow〕

需要 PyTorch 或 TensorFlow。

```python
import torch
import tensorflow as tf
# Torch 每个元素转换为一条记录。
torch_source = torch.utils.data.TensorDataset(torch.arange(4))
print(ray.data.from_torch(torch_source).take(1))
# TensorFlow Dataset 的元素使用字典保留字段名。
tf_source = tf.data.Dataset.from_tensor_slices({"x": [1, 2]})
print(ray.data.from_tf(tf_source).take(1))
```

### 019. from_dask / from_modin / from_mars / from_spark / from_daft〔外部 DataFrame 模板〕

分别需要对应库及其运行环境，下面变量由调用方提供，不要作为本地脚本直接运行。

```python
# 这些函数接收对应库的 DataFrame，而非文件路径。
a = ray.data.from_dask(dask_dataframe)
b = ray.data.from_modin(modin_dataframe)
c = ray.data.from_mars(mars_dataframe)
d = ray.data.from_spark(spark_dataframe)
e = ray.data.from_daft(daft_dataframe)
# 任选一个消费验证；按实际环境准备对应的输入对象。
print(a.take(1))
```

### 020. read_sql：DB-API 数据库〔本地〕

用 SQLite 练习；分布式环境需共享文件路径或实际数据库连接。

```python
import sqlite3
# 连接工厂在执行读取的 Worker 中调用，不把已打开连接序列化过去。
db_path = str(ROOT / "demo.sqlite")
with sqlite3.connect(db_path) as connection:
    pdf.to_sql("records", connection, index=False, if_exists="replace")
def connect_db():
    return sqlite3.connect(db_path)
print(ray.data.read_sql("SELECT id, score FROM records", connect_db).take(2))
```

### 021. read_mongo：MongoDB〔数据库〕

需要 pymongoarrow、网络和实际数据库。

```python
# pipeline 在 MongoDB 一侧筛选数据。
mongo = ray.data.read_mongo(
    uri="mongodb://mongo:27017", database="demo", collection="records",
    pipeline=[{"$match": {"active": True}}],
)
print(mongo.take(1))
```

### 022. read_bigquery / read_databricks_tables / read_iceberg / read_hive〔数据库模板〕

服务连接器版本变化较快；每项展示其入口，具体认证参数按当前 API 配置。本机发布版未导出 `read_hive`，此项以较新文档和仓库源码为准，不能直接在该 wheel 中运行。

```python
# BigQuery：项目身份须有查询与读取权限。
bq = ray.data.read_bigquery(project_id="my-project", dataset="demo.records")
# Databricks：按文档配置访问身份。
db = ray.data.read_databricks_tables(
    warehouse_id="warehouse-id", catalog="main", schema="demo", table="records"
)
# Iceberg：提供已配置的 PyIceberg catalog。
catalog_kwargs = {"name": "demo", "type": "rest", "uri": "https://catalog.example.com"}
iceberg = ray.data.read_iceberg(table_identifier="demo.records", catalog_kwargs=catalog_kwargs)
# Hive：提供真实 SQL 与连接配置；参数请按安装版本核对。
from impala.dbapi import connect
# 工厂在 Worker 执行；真实环境配置 TLS 和认证。
def connect_hive():
    return connect(host="hive-server", port=10000)
hive = ray.data.read_hive(query="SELECT * FROM demo.records", connection_factory=connect_hive)
print(bq.take(1))
```

### 023. read_kafka：一次性消费有界区间〔Kafka〕

需要 Kafka 连接器、Broker、Topic 和安装版本支持。

```python
# trigger=once 是有界数据读取，不是 Flink 式永久运行的流作业。
kafka = ray.data.read_kafka(
    "events", bootstrap_servers="broker:9092",
    start_offset="earliest", end_offset="latest", trigger="once",
)
print(kafka.take(1))
```

<a id="section-2"></a>
## 2. 检查数据与执行结果

官方章节：[查看原文](https://docs.ray.io/en/latest/data/inspecting-data.html)。

### 024. schema〔本地〕

查看列名和类型；可能需要读取部分数据。

```python
# 查看列名和类型；可能需要读取部分数据。
print(ds.schema())
```

### 025. count〔本地〕

查询行数；可能利用元数据，不代表 UDF 一定执行过。

```python
# 查询行数；可能利用元数据，不代表 UDF 一定执行过。
assert ds.count() == 6
```

### 026. show〔本地〕

打印前几行，适合交互检查。

```python
# 打印前几行，适合交互检查。
ds.show(limit=2)
```

### 027. take〔本地〕

只返回指定数量，底层仍可能处理更多行。

```python
# 只返回指定数量，底层仍可能处理更多行。
assert len(ds.take(2)) == 2
```

### 028. take_all〔本地〕

将全部数据拉回 Driver，只用于小数据。

```python
# 将全部数据拉回 Driver，只用于小数据。
assert len(ds.take_all()) == 6
```

### 029. take_batch〔本地〕

按批取样，可指定格式。

```python
# 按批取样，可指定格式。
print(ds.take_batch(2, batch_format="numpy"))
```

### 030. columns〔本地〕

查看字段名。

```python
# 查看字段名。
print(ds.columns())
```

### 031. materialize〔本地〕

显式执行并保留返回值；不等于持久化 Checkpoint。

```python
# 显式执行并保留返回值；不等于持久化 Checkpoint。
cached = ds.map(lambda r: {**r, "score": r["score"] * 2}).materialize()
assert cached.count() == 6
```

### 032. stats〔本地〕

先执行，再查看统计；不替代实时监控。

```python
# 先执行，再查看统计；不替代实时监控。
cached = ds.materialize()
print(cached.stats())
```

<a id="section-3"></a>
## 3. 逐行、批次和字段转换

官方章节：[查看原文](https://docs.ray.io/en/latest/data/transforming-data.html)。

### 033. map：逐行转换〔本地〕

一个用户函数调用处理一行，不等于每行一个 Task。

```python
# 返回新的行字典，保留已有字段。
result = ds.map(lambda row: {**row, "text": row["text"].strip()})
assert result.take(1)[0]["text"] == "ray data 0"
```

### 034. filter：过滤行〔本地〕

只保留条件为真的记录。

```python
# score 为 60、70、80、90 的记录被保留。
result = ds.filter(lambda row: row["score"] >= 60)
assert result.count() == 4
```

### 035. flat_map：一行拆多行〔本地〕

返回的列表可以为空，也可以包含多条记录。

```python
# 将每条文本拆成单词，每个单词输出一行。
result = ds.flat_map(lambda row: [{"id": row["id"], "word": w} for w in row["text"].split()])
print(result.take(3))
```

### 036. map_batches：NumPy 向量化〔本地〕

输入是列数组字典，避免原地修改可能只读的输入缓冲。

```python
def normalize(batch):
    # 新列通过向量化计算生成，不在 Python 中逐行乘除。
    return {**batch, "score": batch["score"] / 100.0}
print(ds.map_batches(normalize, batch_format="numpy", batch_size=2).take(2))
```

### 037. map_batches：Pandas 格式〔本地〕

适合现有 Pandas 逻辑。

```python
def pandas_transform(batch):
    # assign 返回新 DataFrame，不依赖可写共享视图。
    return batch.assign(score=batch["score"] + 1)
print(ds.map_batches(pandas_transform, batch_format="pandas").take(2))
```

### 038. map_batches：PyArrow 格式〔本地〕

列运算交给 Arrow Compute。

```python
import pyarrow.compute as pc
def arrow_transform(batch):
    # 在 Arrow Table 上添加一列。
    return batch.append_column("double_score", pc.multiply(batch["score"], 2))
print(ds.map_batches(arrow_transform, batch_format="pyarrow").take(2))
```

### 039. 可调用类：Actor 状态复用〔本地〕

昂贵初始化放在 __init__，不是每批重新加载。

```python
class AddOffset:
    def __init__(self, offset):
        self.offset = offset  # 模拟模型或查表的初始化状态。
    def __call__(self, batch):
        return {**batch, "score": batch["score"] + self.offset}
result = ds.map_batches(
    AddOffset, fn_constructor_args=(10,), batch_format="numpy",
    compute=ray.data.ActorPoolStrategy(size=1),
)
print(result.take(2))
```

### 040. CPU / 内存与固定池〔本地〕

memory 是调度需求，不是操作系统硬上限。

```python
# 每个 Actor 请求 1 CPU 和 128 MiB 调度内存，最多两个实例。
class Identity:
    def __call__(self, batch):
        return batch
result = ds.map_batches(
    Identity, batch_size=2, num_cpus=1, memory=128 * 1024**2,
    compute=ray.data.ActorPoolStrategy(size=2),
)
print(result.take(1))
```

### 041. GPU 批量计算〔GPU〕

需要 CUDA 版 PyTorch 和可用 GPU；num_gpus 不保证显存隔离。

```python
import torch
class GPUScale:
    def __call__(self, batch):
        # Ray 分配可见设备，用户显式把数据送入 CUDA。
        x = torch.as_tensor(batch["score"], device="cuda")
        return {"score": (x * 2).cpu().numpy()}
result = ds.map_batches(
    GPUScale, num_gpus=1, batch_format="numpy",
    compute=ray.data.ActorPoolStrategy(size=1),
)
print(result.take(2))
```

### 042. 弹性 Actor 池〔本地〕

上限不是承诺同时存在足够资源。

```python
class Identity:
    def __call__(self, batch):
        return batch
# 允许 1 到 2 个 Actor，按资源和执行需求扩展。
result = ds.map_batches(Identity, compute=ray.data.ActorPoolStrategy(min_size=1, max_size=2))
print(result.take(1))
```

### 043. 执行顺序 preserve_order〔本地〕

保序可能影响吞吐；显式排序才表达按字段排序。

```python
from ray.data import DataContext
ctx = DataContext.get_current()
old = ctx.execution_options.preserve_order
try:
    ctx.execution_options.preserve_order = True  # 在创建与执行 Dataset 前设置。
    ordered = ray.data.range(6).map(lambda r: {"id": r["id"] * 2})
    print(ordered.take_all())
finally:
    ctx.execution_options.preserve_order = old  # 避免影响其他例子。
```

### 044. select_columns〔本地〕

投影只保留需要的列。

```python
# 投影只保留需要的列。
result = ds.select_columns(["id", "score"])
print(result.take(2))
```

### 045. drop_columns〔本地〕

移除不用的字段。

```python
# 移除不用的字段。
result = ds.drop_columns(["text"])
print(result.take(2))
```

### 046. rename_columns〔本地〕

重命名字段，其他字段保留。

```python
# 重命名字段，其他字段保留。
result = ds.rename_columns({"score": "rating"})
print(result.take(2))
```

### 047. add_column〔本地〕

以 Pandas 批次函数新增列。

```python
# 以 Pandas 批次函数新增列。
result = ds.add_column("passed", lambda batch: batch["score"] >= 60)
print(result.take(2))
```

### 048. with_column：表达式〔本地〕

当前表达式 API 为 Alpha；避免 Python 逐行计算。

```python
# 当前表达式 API 为 Alpha；避免 Python 逐行计算。
from ray.data.expressions import col
result = ds.with_column("double_score", col("score") * 2)
print(result.take(2))
```

### 049. filter：表达式〔本地〕

布尔表达式使用 & / |，不要写 Python and / or。

```python
# 布尔表达式使用 & / |，不要写 Python and / or。
from ray.data.expressions import col
result = ds.filter(expr=(col("score") >= 60) & (col("id") < 5))
print(result.take(2))
```

### 050. map_groups：整组转换〔本地〕

同一组必须能被该转换处理，热 Key 可能造成内存压力。

```python
def center_group(batch):
    # 输入是同一 group 的 Pandas DataFrame。
    return batch.assign(centered=batch["score"] - batch["score"].mean())
result = ds.groupby("group").map_groups(center_group, batch_format="pandas")
print(result.take_all())
```

<a id="section-4"></a>
## 4. 聚合统计

官方章节：[查看原文](https://docs.ray.io/en/latest/data/aggregating-data.html)。

### 051. sum〔本地〕

求和

```python
# 对整列聚合，返回标量；数据应可表示为数值。
print(ds.sum("score"))
```

### 052. min〔本地〕

最小值

```python
# 对整列聚合，返回标量；数据应可表示为数值。
print(ds.min("score"))
```

### 053. max〔本地〕

最大值

```python
# 对整列聚合，返回标量；数据应可表示为数值。
print(ds.max("score"))
```

### 054. mean〔本地〕

均值

```python
# 对整列聚合，返回标量；数据应可表示为数值。
print(ds.mean("score"))
```

### 055. std〔本地〕

标准差，注意自由度约定

```python
# 对整列聚合，返回标量；数据应可表示为数值。
print(ds.std("score"))
```

### 056. groupby().count：分组计数〔本地〕

每个组生成一行聚合结果。

```python
# A、B 各有三行；聚合后排序仅用于稳定展示。
print(ds.groupby("group").count().sort("group").take_all())
```

### 057. groupby().sum：分组求和〔本地〕

其他 min/max/mean/std 可使用相同分组形式。

```python
# 不把整组拉到 Driver，使用内置聚合执行分布式归并。
print(ds.groupby("group").sum("score").take_all())
```

### 058. aggregate：一次执行多项统计〔本地〕

比为每一项分别扫描更合适。

```python
from ray.data.aggregate import Count, Mean, Min, Max, Std
# 一个聚合计划同时计算这些指标。
print(ds.aggregate(Count(), Mean("score"), Min("score"), Max("score"), Std("score")))
```

### 059. Quantile：分位数〔本地〕

按当前 API 的分位数定义计算。

```python
from ray.data.aggregate import Quantile
# q 在 0 到 1 之间；这里查询 75% 分位。
print(ds.aggregate(Quantile(on="score", q=0.75)))
```

### 060. groupby().aggregate：多项分组统计〔本地〕

每组同时输出计数和均值。

```python
from ray.data.aggregate import Count, Mean
print(ds.groupby("group").aggregate(Count(), Mean("score")).take_all())
```

### 061. AggregateFnV2：自定义聚合〔本地〕

用平方和展示分块局部计算与结果合并，输入为非空数值。

```python
from ray.data.aggregate import AggregateFnV2
from ray.data.block import BlockAccessor
class SumSquares(AggregateFnV2):
    def __init__(self):
        super().__init__(name="sum_squares", on="score", ignore_nulls=True, zero_factory=lambda: 0.0)
    def aggregate_block(self, block):
        values = BlockAccessor.for_block(block).to_numpy()["score"]
        return float(np.square(values).sum())  # 每个 Block 的局部值。
    def combine(self, left, right):
        return left + right  # 结合运算；不能依赖块的到达顺序。
print(ds.aggregate(SumSquares()))
```

<a id="section-5"></a>
## 5. 迭代、切分与消费

官方章节：[查看原文](https://docs.ray.io/en/latest/data/iterating-over-data.html)。

### 062. iter_rows：逐行消费〔本地〕

流式拉取到调用方，不一次收集全部数据。

```python
# 适合逐条发送到下游，Python 循环在当前进程执行。
for row in ds.iter_rows():
    print(row["id"], row["score"])
```

### 063. iter_batches：按批消费〔本地〕

控制调用方的批次大小与格式。

```python
# 遍历所有批次，最后一个批次可能不足 4 行。
for batch in ds.iter_batches(batch_size=4, batch_format="numpy", prefetch_batches=1):
    print(batch["id"].shape)
```

### 064. 迭代时本地缓冲洗牌〔本地〕

只在缓冲范围随机化，不是全局 Shuffle。

```python
# 缓冲行数至少为 batch_size；种子便于可重复实验。
for batch in ds.iter_batches(batch_size=2, local_shuffle_buffer_size=4, local_shuffle_seed=42):
    print(batch["id"])
```

### 065. streaming_split：分布式同步消费〔本地〕

多个分片必须并行消费；依次在 Driver 耗尽各分片可能阻塞。

```python
@ray.remote
def consume(shard):
    return sum(len(batch["id"]) for batch in shard.iter_batches(batch_size=2))
# DataIterator 不需要先物化整个数据集。
shards = ds.streaming_split(2, equal=True)
print(ray.get([consume.remote(shard) for shard in shards]))
```

### 066. split：物化分片〔本地〕

适合小数据；不同于 streaming_split。

```python
# split 返回多个 MaterializedDataset，equal=True 会尽量等量切分。
parts = ds.split(2, equal=True)
assert [part.count() for part in parts] == [3, 3]
```

### 067. train_test_split：训练测试划分〔本地〕

种子用于随机切分；不要把它当成严格分层抽样。

```python
# 示例只演示切分，样本太少时比例会有取整。
train, test = ds.train_test_split(test_size=0.3, shuffle=True, seed=42)
print(train.count(), test.count())
```

### 068. limit：截取少量数据〔本地〕

适合调试，不是随机采样。

```python
# 只保留最多 3 行。
assert ds.limit(3).count() == 3
```

### 069. random_sample：概率采样〔本地〕

fraction 是每行采样概率，不保证精确样本数。

```python
# 小样本中结果可能为空，不要断言一定恰好三行。
print(ds.random_sample(0.5, seed=42).take_all())
```

<a id="section-6"></a>
## 6. Join：数据连接

官方章节：[查看原文](https://docs.ray.io/en/latest/data/joining-data.html)。

### 070. inner〔本地〕

只保留两边匹配的记录；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="inner", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 071. left_outer〔本地〕

保留全部左行，右侧缺失填空值；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="left_outer", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 072. right_outer〔本地〕

保留全部右行；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="right_outer", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 073. full_outer〔本地〕

保留两边全部键；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="full_outer", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 074. left_semi〔本地〕

只返回在右边存在匹配的左行；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="left_semi", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 075. left_anti〔本地〕

只返回在右边没有匹配的左行；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="left_anti", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 076. right_semi〔本地〕

只返回在左边有匹配的右行；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="right_semi", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 077. right_anti〔本地〕

只返回在左边没有匹配的右行；Join 需要 Polars，当前仍需注意实验性行为。

```python
# 左键为 0..5，右键为 1、8，列名刻意避免与左表冲突。
right = ray.data.from_items([{"id": 1, "city": "上海"}, {"id": 8, "city": "北京"}])
joined = ds.join(right, join_type="right_anti", on=("id",), num_partitions=2)
print(joined.take_all())
```

### 078. 不同键名连接与后缀〔本地〕

on 与 right_on 显式指定两侧键。

```python
right = ray.data.from_items([{"uid": 1, "score": 99.0}])
# 同名非键字段用后缀区分。
joined = ds.join(
    right, join_type="inner", on=("id",), right_on=("uid",),
    num_partitions=2, left_suffix="_left", right_suffix="_right",
)
print(joined.take_all())
```

### 079. Join 分区与 Aggregator 配置〔本地〕

大分区有 OOM 风险，过多小分区增加开销。

```python
from ray.data import DataContext
ctx = DataContext.get_current()
old = ctx.max_hash_shuffle_aggregators
try:
    ctx.max_hash_shuffle_aggregators = 2  # 不会凭空增加集群 CPU。
    right = ray.data.from_items([{"id": 1, "flag": True}])
    print(ds.join(right, join_type="inner", num_partitions=2, on=("id",)).take_all())
finally:
    ctx.max_hash_shuffle_aggregators = old
```

<a id="section-7"></a>
## 7. Shuffle 与重分区

官方章节：[查看原文](https://docs.ray.io/en/latest/data/shuffling-data.html)。

### 080. 读取时文件洗牌〔外部存储〕

随机文件顺序不是全局行洗牌；为真实多文件输入替换路径。

```python
# 适合一个目录内有很多文件的情况。
loaded = ray.data.read_parquet("s3://bucket/input/", shuffle="files")
print(loaded.take(2))
```

### 081. randomize_block_order：块顺序随机化〔本地〕

块内部的行不因此随机化，通常比全局洗牌便宜。

```python
# 固定 seed 便于调试。
print(ds.randomize_block_order(seed=42).take_all())
```

### 082. random_shuffle：全局行洗牌〔本地〕

跨块交换数据，比本地缓冲洗牌更重。

```python
# 完整消费才看到全局结果；大数据不要用 take_all。
print(ds.random_shuffle(seed=42).take_all())
```

### 083. map_batches 内洗牌〔本地〕

只随机当前批次，不保证全数据集随机。

```python
def shuffle_batch(batch):
    index = np.random.default_rng(42).permutation(len(batch["id"]))
    return {name: values[index] for name, values in batch.items()}
print(ds.map_batches(shuffle_batch, batch_format="numpy", batch_size=4).take_all())
```

### 084. repartition：调整 Block 数〔本地〕

默认不做全局随机洗牌。

```python
# 这里改变分片数量；不保证每个 Block 字节数一致。
result = ds.repartition(3).materialize()
assert result.num_blocks() == 3
```

### 085. repartition(shuffle=True)：重分布〔本地〕

涉及跨分片交换；不是免费调整元数据。

```python
# 将输入重新分到两个 Block。
print(ds.repartition(2, shuffle=True).take_all())
```

### 086. 按目标行数流式重分块〔本地〕

当前 API 支持 target_num_rows_per_block；固定行数不等于固定内存。

```python
# 尽量让每个输出 Block 两行；末尾块可较小。
print(ds.repartition(target_num_rows_per_block=2).take_all())
```

### 087. sort：按字段排序〔本地〕

排序会产生数据交换和协调。

```python
# 全局按 score 降序。
assert ds.sort("score", descending=True).take(1)[0]["score"] == 90.0
```

<a id="section-8"></a>
## 8. 加权数据集混合

官方章节：[查看原文](https://docs.ray.io/en/latest/data/mixing-data.html)。

### 088. mix：按权重混合〔本地〕

输入 Schema 需一致；预处理放在 mix 前。

```python
left = ray.data.from_items([{"x": "A"}] * 6).repartition(3)
right = ray.data.from_items([{"x": "B"}] * 2).repartition(1)
# 基于 Block 混合，比例不是每个小批次都精确相等。
mixed = left.mix(right, weights=[0.75, 0.25])
print(list(mixed.iter_batches(batch_size=4)))
```

### 089. mix：停止条件〔本地〕

较短输入耗尽后的行为需要显式选择。

```python
from ray.data import MixStoppingCondition
left = ray.data.from_items([{"x": 1}] * 6)
right = ray.data.from_items([{"x": 2}] * 2)
# 任一输入耗尽就停止，可能留下未消费数据。
mixed = left.mix(right, weights=[0.5, 0.5], stopping_condition=MixStoppingCondition.STOP_ON_SHORTEST)
print(list(mixed.iter_batches(batch_size=2)))
```

<a id="section-9"></a>
## 9. 写出与转换到其他库

官方章节：[查看原文](https://docs.ray.io/en/latest/data/saving-data.html)。

### 090. write_parquet〔本地〕

写出到独立目录；默认文件数取决于分块及写出策略。

```python
# ROOT 是本次本地练习的临时目录，目录不会覆盖用户数据。
path = str(ROOT / "output-parquet")
ds.write_parquet(path)
assert ray.data.read_parquet(path).count() == 6
```

### 091. write_csv〔本地〕

写出到独立目录；默认文件数取决于分块及写出策略。

```python
# ROOT 是本次本地练习的临时目录，目录不会覆盖用户数据。
path = str(ROOT / "output-csv")
ds.write_csv(path)
assert ray.data.read_csv(path).count() == 6
```

### 092. write_json〔本地〕

写出到独立目录；默认文件数取决于分块及写出策略。

```python
# ROOT 是本次本地练习的临时目录，目录不会覆盖用户数据。
path = str(ROOT / "output-json")
ds.write_json(path)
assert ray.data.read_json(path).count() == 6
```

### 093. S3 写出〔外部存储〕

同一逻辑任务重跑需独立考虑提交与幂等。

```python
# append 不会自动去重；不要把路径当作文件名。
ds.write_parquet("s3://bucket/output/run-001/", mode="append")
```

### 094. 控制输出文件规模〔本地〕

Block 数与最终文件数并非永远一一对应。

```python
# 调整分块，并给 Parquet 写入设置每文件行数目标。
ds.repartition(2).write_parquet(str(ROOT / "sized-output"), min_rows_per_file=2)
```

### 095. 分区目录写出〔本地〕

适用于按列值裁剪读取。

```python
# 输出按 group 字段创建分区目录。
ds.write_parquet(str(ROOT / "partitioned"), partition_cols=["group"])
print(ray.data.read_parquet(str(ROOT / "partitioned")).take(1))
```

### 096. to_pandas：收集到 Driver〔本地〕

只对能放入单机内存的小数据使用。

```python
# 不要对 TB 数据无约束调用。
frame = ds.to_pandas()
assert len(frame) == 6
```

### 097. to_numpy_refs：导出分布式数组引用〔本地〕

只有 ray.get 才把对应数据取到调用方。

```python
# 本地小数据可演示全部读取；大数据按需消费。
refs = ds.select_columns(["id", "score"]).to_numpy_refs()
print([batch["id"].shape for batch in ray.get(refs)])
```

### 098. to_arrow_refs：导出 Arrow 引用〔本地〕

按分块导出，不等于将全部结果先合并。

```python
refs = ds.to_arrow_refs()
# 本例仅获取第一个块，避免不必要的集中收集。
print(ray.get(refs[0]).schema)
```

### 099. to_dask / to_modin / to_mars / to_spark / to_daft〔外部 DataFrame 模板〕

分别需要对应库及会话；示例变量由调用方准备。

```python
# 一次只使用与实际环境对应的转换。
dask_frame = ds.to_dask()
modin_frame = ds.to_modin()
mars_frame = ds.to_mars()
spark_frame = ds.to_spark(spark)  # spark 为已创建的 SparkSession。
daft_frame = ds.to_daft()
# 各框架后续执行方式不同，不要因此假设全部收集到 Driver。
print(dask_frame.head())
```

<a id="section-10"></a>
## 10. 图像数据操作

官方章节：[查看原文](https://docs.ray.io/en/latest/data/working-with-images.html)。

### 100. 创建、读取图像〔Pillow〕

需要 Pillow；本例生成小 PNG，不下载图像。

```python
from PIL import Image
# 在单机练习目录生成两张图片。
image_dir = ROOT / "images"
image_dir.mkdir(exist_ok=True)
for i in range(2):
    Image.fromarray(np.full((16, 16, 3), i * 100, dtype=np.uint8)).save(image_dir / f"{i}.png")
images = ray.data.read_images(str(image_dir), mode="RGB")
print(images.take_batch(2)["image"].shape)
```

### 101. 图像 resize 与归一化〔Pillow〕

先运行前一个图像准备示例。

```python
from PIL import Image
images = ray.data.read_images(str(ROOT / "images"), mode="RGB")
def resize(row):
    # 改为相同形状，便于批量堆叠。
    return {"image": np.asarray(Image.fromarray(row["image"]).resize((8, 8)))}
resized = images.map(resize)
scaled = resized.map_batches(
    lambda b: {"image": b["image"].astype(np.float32) / 255}, batch_format="numpy"
)
print(scaled.take_batch(2)["image"].shape)
```

### 102. 图像批量推理〔Pillow〕

用亮度阈值分类演示批处理；不是训练好的识别模型。

```python
images = ray.data.read_images(str(ROOT / "images"), mode="RGB")
class BrightnessClassifier:
    def __call__(self, batch):
        # 计算每张图像平均亮度并输出二分类标签。
        return {"label": (batch["image"].mean(axis=(1, 2, 3)) > 50).astype(np.int64)}
print(images.map_batches(BrightnessClassifier, batch_format="numpy",
    compute=ray.data.ActorPoolStrategy(size=1)).take_all())
```

### 103. write_images：保存图像〔Pillow〕

需先准备 ROOT/images；写出要求列内容为图像数组。

```python
images = ray.data.read_images(str(ROOT / "images"), mode="RGB")
# column 必须与数据集中的图像列一致。
images.write_images(str(ROOT / "image-output"), column="image", file_format="png")
```

<a id="section-11"></a>
## 11. 文本数据操作

官方章节：[查看原文](https://docs.ray.io/en/latest/data/working-with-text.html)。

### 104. 文本清洗与分句〔本地〕

使用本地文本文件，避免模型下载。

```python
texts = ray.data.read_text(str(ROOT / "input.txt"))
# 先去除空白和转小写，再按空格拆成多行。
clean = texts.map(lambda r: {"text": r["text"].strip().lower()})
words = clean.flat_map(lambda r: [{"word": w} for w in r["text"].split()])
print(words.take_all())
```

### 105. Tokenize：复用 Tokenizer〔Transformers〕

需要 transformers、模型访问；与目标模型使用同一 tokenizer。

```python
from transformers import AutoTokenizer
class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    def __call__(self, batch):
        # 输出变长列表，避免无意固定 Padding 到最大长度。
        out = self.tokenizer(batch["text"].tolist(), truncation=True, max_length=128)
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}
result = ds.select_columns(["text"]).map_batches(
    Tokenizer, batch_format="numpy", compute=ray.data.ActorPoolStrategy(size=1)
)
print(result.take(1))
```

### 106. 文本推理与保存〔本地〕

用规则分类演示数据路径；换模型时将初始化放在 Actor 内。

```python
# 原生规则可以直接逐行运行，无需初始化大模型。
classified = ds.map(lambda r: {"id": r["id"], "label": "ray" if "ray" in r["text"] else "other"})
classified.write_parquet(str(ROOT / "text-classified"))
```

<a id="section-12"></a>
## 12. 张量与 NumPy 操作

官方章节：[查看原文](https://docs.ray.io/en/latest/data/working-with-tensors.html)。

### 107. 定长张量批次〔本地〕

同形状记录可以堆成 (batch, ...) 数组。

```python
tensors = ray.data.from_numpy(np.arange(24).reshape(6, 2, 2))
# 2 个样本，每个样本 2×2。
assert tensors.take_batch(2)["data"].shape == (2, 2, 2)
```

### 108. 变长张量〔本地〕

不同形状通常不能直接 np.stack，应先逐样本处理或 Padding。

```python
ragged = ray.data.from_items([{"tensor": np.ones((n, 2))} for n in [2, 3]])
def lengths(batch):
    # 不把不同长度的张量强行堆叠成规则三维数组。
    return {"length": np.array([len(x) for x in batch["tensor"]])}
print(ragged.map_batches(lengths, batch_format="numpy").take_all())
```

### 109. 批量张量变换与保存〔本地〕

Parquet 可保存 Ray 张量扩展类型，跨库读取需核对兼容性。

```python
tensors = ray.data.from_numpy(np.arange(12).reshape(6, 2))
scaled = tensors.map_batches(lambda b: {"data": np.clip(b["data"], 0, 5)}, batch_format="numpy")
scaled.write_parquet(str(ROOT / "tensor-parquet"))
# NumPy 写出按列保存，生成多个 .npy 分片。
scaled.write_numpy(str(ROOT / "tensor-numpy"), column="data")
```

<a id="section-13"></a>
## 13. PyTorch 训练与批量推理

官方章节：[查看原文](https://docs.ray.io/en/latest/data/working-with-pytorch.html)。

### 110. iter_torch_batches：训练输入〔PyTorch〕

需要 torch；数据加载不是模型训练本身。

```python
# 特征列转为 Torch Tensor，避免逐行 Python 数据加载。
for batch in ds.select_columns(["id", "score"]).iter_torch_batches(batch_size=2):
    print(batch["score"].shape, batch["score"].dtype)
```

### 111. UDF 使用 Torch 并返回 NumPy〔PyTorch〕

输出移回 CPU，适合通用写出。

```python
import torch
def torch_transform(batch):
    # 将数值列送入 Torch；此例使用 CPU。
    tensor = torch.as_tensor(batch["score"].copy(), dtype=torch.float32)
    return {"score": (tensor * 2).numpy()}
print(ds.map_batches(torch_transform, batch_format="numpy").take(2))
```

### 112. 批量推理与模型复用〔PyTorch〕

演示模型形状和结果消费；随机线性层不具有业务精度。

```python
import torch
class Predictor:
    def __init__(self):
        self.model = torch.nn.Linear(1, 2).eval()
    def __call__(self, batch):
        x = torch.as_tensor(batch["score"].copy(), dtype=torch.float32).reshape(-1, 1)
        with torch.inference_mode():
            return {"prediction": self.model(x).argmax(dim=1).numpy()}
print(ds.map_batches(Predictor, batch_format="numpy",
    compute=ray.data.ActorPoolStrategy(size=1)).take(2))
```

### 113. to_tf：供 TensorFlow 训练〔TensorFlow〕

feature / label 列必须与训练器输入一致。

```python
# TensorFlow 需要安装；这里用 id 充当示例标签，不代表实际任务设计。
tf_dataset = ds.to_tf(feature_columns="score", label_columns="id", batch_size=2)
for features, labels in tf_dataset.take(1):
    print(features.shape, labels.shape)
```

### 114. Ray Train 分片消费〔Ray Train / PyTorch〕

需要 ray[train] 与 torch，资源足够运行两个训练 Worker。

```python
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import ray.train

def train_loop(config):
    # 在每个训练 Worker 取得自己的输入分片，不要在外部逐个串行消费。
    shard = ray.train.get_dataset_shard("train")
    total = 0
    for batch in shard.iter_torch_batches(batch_size=2):
        total += len(batch["id"])
    ray.train.report({"rows": total})

trainer = TorchTrainer(train_loop_per_worker=train_loop,
    scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    datasets={"train": ds.select_columns(["id", "score"])})
print(trainer.fit().metrics)
```

<a id="section-14"></a>
## 14. LLM Processor 数据操作

官方章节：[查看原文](https://docs.ray.io/en/latest/data/working-with-llms.html)。

### 115. vLLM 文本生成〔GPU / vLLM〕

每条输入准备 messages 和采样配置，消费输出触发推理。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
# 需要兼容 vLLM、CUDA、可用 GPU 和模型下载权限。
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-0.5B-Instruct", batch_size=4, concurrency=1,
    engine_kwargs={"max_model_len": 512},
)
processor = build_processor(config,
    preprocess=lambda r: {"messages": [{"role": "user", "content": r["text"]}],
                          "sampling_params": {"temperature": 0, "max_tokens": 32}},
    postprocess=lambda r: {"response": r["generated_text"]})
print(processor(ds.select_columns(["text"])).take(1))
```

### 116. Embedding 向量生成〔GPU / vLLM〕

模型必须支持 embedding 任务。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
# 需要兼容 vLLM、CUDA、可用 GPU 和模型下载权限。
config = vLLMEngineProcessorConfig(
    model_source="sentence-transformers/all-MiniLM-L6-v2", task_type="embed",
    batch_size=4, concurrency=1, chat_template_stage=False, detokenize_stage=False,
    engine_kwargs={"max_model_len": 256, "enable_prefix_caching": False,
                   "enable_chunked_prefill": False},
)
processor = build_processor(config, preprocess=lambda r: {"prompt": r["text"]},
    postprocess=lambda r: {"embedding": r["embeddings"]})
print(processor(ds.select_columns(["text"])).take(1))
```

### 117. 分类模型〔GPU / vLLM〕

使用分类而非自由文本生成接口；输出含义由模型决定。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
# 需要兼容 vLLM、CUDA、可用 GPU 和模型下载权限。
config = vLLMEngineProcessorConfig(
    model_source="nvidia/nemocurator-fineweb-nemotron-4-edu-classifier",
    task_type="classify", batch_size=4, concurrency=1,
    chat_template_stage=False, detokenize_stage=False,
)
processor = build_processor(config, preprocess=lambda r: {"prompt": r["text"]},
    postprocess=lambda r: {"scores": r["embeddings"]})
print(processor(ds.select_columns(["text"])).take(1))
```

### 118. OpenAI 兼容端点〔外部推理服务〕

使用调用方配置的端点与身份，不将令牌写进文件。

```python
import os
from ray.data.llm import HttpRequestProcessorConfig, build_processor
config = HttpRequestProcessorConfig(
    url=os.environ["LLM_CHAT_URL"],  # 完整的 /v1/chat/completions URL。
    headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]}, qps=1,
)
processor = build_processor(config,
    preprocess=lambda r: {"payload": {"model": os.environ["LLM_MODEL"],
        "messages": [{"role": "user", "content": r["text"]}], "max_tokens": 32}},
    postprocess=lambda r: {"response": r["http_response"]["choices"][0]["message"]["content"]})
print(processor(ds.select_columns(["text"])).take(1))
```

### 119. 多模态图像输入〔GPU / 多模态模型〕

图像 URL、模型和输入内容必须匹配；替换为真实可访问资源。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
# 需要兼容 vLLM、CUDA、可用 GPU 和模型下载权限。
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct", concurrency=1, batch_size=1,
    prepare_multimodal_stage=True,
)
processor = build_processor(config,
    preprocess=lambda r: {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": r["image_url"]}},
    ]}], "sampling_params": {"max_tokens": 64}},
    postprocess=lambda r: {"response": r["generated_text"]})
source = ray.data.from_items([{"image_url": "https://your-host/image.png"}])
print(processor(source).take(1))
```

### 120. Tokenize / Detokenize 阶段拆分〔GPU / vLLM〕

Ray Data LLM 可以在独立 CPU 阶段完成分词与解码。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
# 需要兼容 vLLM、CUDA、可用 GPU 和模型下载权限。
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-0.5B-Instruct", concurrency=1, batch_size=4,
    tokenize_stage=True, detokenize_stage=True,
)
processor = build_processor(config,
    preprocess=lambda r: {"messages": [{"role": "user", "content": r["text"]}],
        "sampling_params": {"max_tokens": 32, "detokenize": False}},
    postprocess=lambda r: {"response": r["generated_text"]})
print(processor(ds.select_columns(["text"])).take(1))
```

### 121. 模型并行与数据并行〔多 GPU 配置〕

每个引擎使用的 GPU 数与引擎副本数分开考虑。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
# 需要兼容 vLLM、CUDA、可用 GPU 和模型下载权限。
# 此例需要满足两个引擎各两张 GPU 的资源请求，且模型适合该配置。
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-7B-Instruct", concurrency=2, batch_size=8,
    engine_kwargs={"tensor_parallel_size": 2, "max_model_len": 1024},
)
# 使用前面的 preprocess / postprocess 模式构建 Processor 后消费。
print(config.model_source)
```

<a id="section-15"></a>
## 15. 内存、监控与执行配置

官方章节：[查看原文](https://docs.ray.io/en/latest/data/execution-configurations.html)。

### 122. DataContext：目标块大小〔本地〕

目标不等于严格上限，也不能取代 Batch 内存管理。

```python
from ray.data import DataContext
ctx = DataContext.get_current()
old = ctx.target_max_block_size
try:
    ctx.target_max_block_size = 32 * 1024**2  # 创建 Pipeline 前设置。
    print(ray.data.range(10).map(lambda r: r).take(1))
finally:
    ctx.target_max_block_size = old
```

### 123. ExecutionOptions：执行资源预算〔本地〕

限制流水线资源使用，不是容器硬限额。

```python
from ray.data import DataContext, ExecutionResources
ctx = DataContext.get_current()
old = ctx.execution_options.resource_limits
try:
    ctx.execution_options.resource_limits = ExecutionResources(cpu=2, object_store_memory=128 * 1024**2)
    print(ray.data.range(10).take(1))
finally:
    ctx.execution_options.resource_limits = old
```

### 124. 详细统计与 OOM 调试〔本地〕

优先区分 Worker Heap、Object Store 与容器限制。

```python
from ray.data import DataContext
ctx = DataContext.get_current()
old = ctx.verbose_stats_logs
try:
    ctx.verbose_stats_logs = True
    # 小 Batch 和受控 Actor 数是起点，具体需测实际工作集。
    cached = ds.map_batches(lambda b: b, batch_size=2).materialize()
    print(cached.stats())
finally:
    ctx.verbose_stats_logs = old
```

### 125. 原生 Checkpoint〔外部存储 / 恢复实验〕

需要当前版本支持、稳定唯一 ID 与可恢复 Sink；不是任意外部副作用的事务。

```python
from ray.data import DataContext
from ray.data.checkpoint import CheckpointConfig
ctx = DataContext.get_current()
ctx.checkpoint_config = CheckpointConfig(
    id_column="id", checkpoint_path="s3://bucket/job-001/checkpoint/",
    delete_checkpoint_on_success=False,
)
# 实际重跑复用同一输入、逻辑版本、输出与 checkpoint，不覆盖已提交文件。
source = ray.data.read_parquet("s3://bucket/job-001/input/")
source.map(lambda r: r).write_parquet("s3://bucket/job-001/output/", mode="append")
```

<a id="section-16"></a>
## 16. 高级转换与自定义文件格式

官方章节：[查看原文](https://docs.ray.io/en/latest/data/custom-datasource-example.html)。

### 126. 异步 UDF〔异步依赖〕

需要文档对应的 uvloop 等依赖；异步推进不等于 CPU 并行。

```python
import asyncio
class AsyncTransform:
    async def __call__(self, batch):
        await asyncio.sleep(0)  # 实际可 await 非阻塞 I/O；不要在此用阻塞请求。
        return batch
print(ds.map_batches(AsyncTransform, compute=ray.data.ActorPoolStrategy(size=1)).take(1))
```

### 127. Placement Group：协调多个 Worker 资源〔高级调度〕

分布式模型需确保子任务放置和资源释放；示例只是资源分组。

```python
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
pg = placement_group([{"CPU": 1}], strategy="PACK")
ray.get(pg.ready())
try:
    # 从 Placement Group 中为该批处理 Task 获取 CPU。
    result = ds.map_batches(lambda b: b, num_cpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg))
    print(result.take(1))
finally:
    remove_placement_group(pg)
```

### 128. 昂贵 collate 移到 Ray Data〔PyTorch〕

需要 PyTorch；将可并行的整理逻辑移到批处理阶段，训练端只消费。

```python
import torch
def collate_in_data(batch):
    # 示例输出规则数值数组；变长样本可在这里 pad/pack。
    x = torch.as_tensor(batch["score"].copy(), dtype=torch.float32).reshape(-1, 1)
    return {"features": x.numpy(), "label": batch["id"]}
prepared = ds.map_batches(collate_in_data, batch_format="numpy", batch_size=2)
for batch in prepared.iter_torch_batches(batch_size=2):
    print(batch["features"].shape)
```

### 129. FileBasedDatasource：自定义读取〔本地〕

每个自定义文件类型实现解析逻辑；本例读取 .txt 为一行记录。

```python
from ray.data.datasource import FileBasedDatasource
class WholeTextSource(FileBasedDatasource):
    def _read_stream(self, file, path):
        # 返回 Arrow Block；这里每个文件是一条记录。
        yield pa.table({"path": [path], "text": [file.readall().decode("utf-8")]})
source = WholeTextSource(str(ROOT / "input.txt"))
print(ray.data.read_datasource(source).take(1))
```

### 130. RowBasedFileDatasink：自定义写出〔本地〕

每行一个文件，适合小文件演示；大数据应考虑块级写出避免小文件过多。

```python
from ray.data.datasource import RowBasedFileDatasink
class WholeTextSink(RowBasedFileDatasink):
    def __init__(self, path):
        super().__init__(path, file_format="txt")
    def write_row_to_file(self, row, file):
        file.write(row["text"].encode("utf-8"))
# 每条记录输出一个文本文件。
ds.select_columns(["text"]).write_datasink(WholeTextSink(str(ROOT / "custom-text")))
```

<a id="section-17"></a>
## 17. 并发数据集与端到端流水线

官方章节：[查看原文](https://docs.ray.io/en/latest/data/concurrent-dataset-execution.html)。

### 131. 子集群标签隔离〔带标签集群 / 版本接口〕

需要已配置 ray-subcluster 标签的节点；无匹配节点会等待。

```python
from concurrent.futures import ThreadPoolExecutor
from ray.data import DataContext

def make_dataset(label):
    # 在主线程先创建各自 Context，避免并发修改全局配置。
    ctx = DataContext.get_current().copy()
    ctx.execution_options.label_selector = {"ray-subcluster": label}
    with DataContext.current(ctx):
        return ray.data.range(10)

left = make_dataset("tenant_a")
right = make_dataset("tenant_b")
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(dataset.materialize) for dataset in [left, right]]
    print([future.result().count() for future in futures])
```

### 132. 端到端：读取 → 处理 → 保存〔本地〕

本地小数据演示完整消费、验证和统计。

```python
# id 便于与输入关联；本例只是规则分类，不调用外部模型。
result = (ray.data.read_parquet(str(ROOT / "input.parquet"))
    .map(lambda r: {**r, "text": r["text"].strip()})
    .filter(lambda r: r["score"] >= 60)
    .map_batches(lambda b: {"id": b["id"], "label": (b["score"] >= 80).astype(np.int64)},
                 batch_size=2, batch_format="numpy"))
path = str(ROOT / "end-to-end")
result.write_parquet(path)
assert ray.data.read_parquet(path).count() == 4
print(result.stats())
```

<a id="section-18"></a>
## 18. 补充：组合、表达式与多模态操作

官方章节：[查看原文](https://docs.ray.io/en/latest/data/transforming-data.html)。

### 133. union：纵向合并〔本地〕

不去重；输入应有兼容 Schema。

```python
# 第二个输入沿用相同 Schema。
combined = ds.union(ds.limit(1))
assert combined.count() == 7
```

### 134. zip：按位置横向组合〔本地〕

需要相同记录数；这是按位置配对，不是按 Key Join。

```python
# 右侧刻意只用新列名，避免重名。
right = ray.data.range(6).map(lambda r: {"weight": r["id"] + 1})
print(ds.zip(right).take(2))
```

### 135. map_batches 生成器：分批输出〔本地〕

适合一次输入产生多个结果批，降低临时输出峰值。

```python
def split_outputs(batch):
    # 每次 yield 一条小批次；生产环境应选择适当粒度，避免过小对象。
    for i in range(len(batch["id"])):
        yield {"id": batch["id"][i:i+1]}
assert ds.map_batches(split_outputs, batch_format="numpy").count() == 6
```

### 136. with_columns：一次投影新增多列〔较新接口〕

较新表达式接口；字段别名通过字典指定。

```python
from ray.data.expressions import col
# 新列都基于原始输入列计算，避免假定可以引用同批新增列。
result = ds.with_columns({"double_score": col("score") * 2, "next_id": col("id") + 1})
print(result.take(2))
```

### 137. udf：表达式中的批量 Arrow 函数〔表达式 Alpha〕

不是逐行 UDF，输入列为 Arrow 数组。

```python
from ray.data.expressions import col, udf
from ray.data.datatype import DataType
import pyarrow.compute as pc
@udf(return_dtype=DataType.float64())
def half_score(values):
    return pc.divide(values, 2.0)  # 按列批量执行。
print(ds.with_column("half", half_score(col("score"))).take(2))
```

### 138. 视频批量推理〔GPU / 视频模型〕

需要兼容 VLM、视频解码依赖、足够显存；URL 必须替换。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen3-VL-4B-Instruct", batch_size=1, concurrency=1,
    engine_kwargs={"limit_mm_per_prompt": {"video": 1}},
    prepare_multimodal_stage={"enabled": True, "chat_template_content_format": "openai"},
)
processor = build_processor(config,
    preprocess=lambda r: {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "概括视频内容"},
        {"type": "video_url", "video_url": {"url": r["url"]}},
    ]}], "sampling_params": {"max_tokens": 64}},
    postprocess=lambda r: {"response": r["generated_text"]})
print(processor(ray.data.from_items([{"url": "https://your-host/video.mp4"}])).take(1))
```

### 139. 音频批量推理〔GPU / 音频模型〕

使用音频理解模型，不是把音频字节作为普通文本分词。

```python
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
config = vLLMEngineProcessorConfig(
    model_source="Qwen/Qwen2.5-Omni-3B", batch_size=1, concurrency=1,
    engine_kwargs={"limit_mm_per_prompt": {"audio": 1}},
    prepare_multimodal_stage={"enabled": True, "chat_template_content_format": "openai"},
)
processor = build_processor(config,
    preprocess=lambda r: {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "描述这段音频"},
        {"type": "audio_url", "audio_url": {"url": r["url"]}},
    ]}], "sampling_params": {"max_tokens": 64}},
    postprocess=lambda r: {"response": r["generated_text"]})
print(processor(ray.data.from_items([{"url": "https://your-host/audio.wav"}])).take(1))
```

## 指南覆盖与阅读边界

| 用户指南主题 | 本文件中的例子 |
| --- | --- |
| Loading / Inspecting / Transforming / Aggregating | 第 1–4 节 |
| Iterating / Joining / Shuffling / Mixing / Saving | 第 5–9 节 |
| Images / Text / Tensors / PyTorch / LLMs | 第 10–14 节 |
| OOM / Monitoring / Execution configurations | 第 15 节及内存、统计例子；系统排查见 `ray-data.md` 第 5、9、10 题 |
| Multiple Datasets / Offline Batch Inference | 第 17 节 |
| Performance / expensive collate / custom files | 第 16 节及批处理、读取投影、资源预算例子 |

第 18 节还补充了视频、音频和表达式操作。LoRA、自定义 Tokenizer、引擎共享属于进一步的配置变体。不要把图像消息体直接改字段就假设任意模型支持，相关 API 和模型支持应按 [LLM 指南](https://docs.ray.io/en/latest/data/working-with-llms.html) 核对。

## 验证记录

- 共 **139 个示例**，全部 Python 片段已通过语法检查。
- 在独立的两 CPU 单机 Ray 会话中实际运行 **80 个本地示例，全部通过**；覆盖本地读取、行/批次转换、表达式、聚合、迭代、部分 Join 类型、Shuffle、Mix、文件写出、张量及端到端处理。
- 环境：Ray 2.59.0、Python 3.12、NumPy 2.5.4、Pandas 3.1.0、PyArrow 26.0.0、Polars 2.1.0。
- 通过的示例编号：001, 002, 003, 004, 005, 006, 008, 009, 010, 011, 012, 013, 015, 020, 024, 025, 026, 027, 028, 029, 030, 031, 032, 033, 034, 035, 036, 037, 038, 039, 044, 045, 046, 047, 048, 049, 050, 051, 052, 053, 054, 055, 056, 057, 058, 059, 060, 061, 062, 063, 064, 068, 069, 070, 071, 075, 081, 082, 083, 084, 085, 086, 087, 088, 090, 091, 092, 095, 096, 097, 098, 104, 106, 107, 108, 109, 132, 133, 134, 135。
- 其余示例仅做语法及接口查阅，**未实际运行**；尤其是外部服务、GPU、模型下载、可选框架、标签集群和故障恢复模板。语法检查不能证明依赖、权限、显存或外部服务可用。
