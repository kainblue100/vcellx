# 数据溯源与完整性核验

本文记录本仓库使用的每一份数据从哪来、如何核验、核验结果是什么。
所有结论都可由 `tools/` 下的脚本复现。

根目录：`bzhang@10.13.8.200:/mnt/storage/bzhang/AIVC`

---

## 一、核验方法

对 Hugging Face 托管的三个数据集，做法是**逐文件比对字节数**而不是只看文件数量：

1. 分页拉取 HF tree API 到拿全清单
   （`tools/fetch_manifest.py`。**HF 的这个接口会分页**，只读第一页会严重低估文件数：
   Tahoe 第一次只返回 993 个文件，实际有 4,427 个）
2. 把清单和磁盘逐路径、逐字节数比对（`tools/audit_local.py`）
3. 报告「一致 / 大小不符（截断）/ 缺失 / 多余」四类

对 cellxgene 的 Tabula Sapiens，做法是**重算 sha256 并与下载记录比对**。

---

## 二、逐数据集结果

### 2.1 Arc Virtual Cell Challenge 训练集

| 项 | 值 |
|---|---|
| 来源 | `huggingface.co/datasets/cyrilzakka/arc-institute-virtual-cell-dataset` |
| 路径 | `arc-institute-virtual-cell-dataset/` |
| 期望 | 5 文件 / 15.48 GB |
| 实测 | **5 一致 / 0 大小不符 / 0 缺失 / 0 多余** |

内容：`adata_Training.h5ad`（15,482.4 MB）、`gene_names.csv`、`pert_counts_Validation.csv`、README、.gitattributes

实际结构（`h5py` 读取）：

```
X            csr_matrix  (221273, 18080)  float32
obs          _index, batch(48 类), guide_id(189 类), target_gene(151 类)
var          _index = 18,080 个基因符号，另有 mean_counts / pct_dropout 等
对照标签      non-targeting，38,176 个细胞
```

### 2.2 Tahoe-100M

| 项 | 值 |
|---|---|
| 来源 | `huggingface.co/datasets/tahoebio/Tahoe-100M` |
| 路径 | `Tahoe-100M/` |
| 期望 | **4,427 文件 / 428.81 GB**（其中 4,419 个 parquet 分片） |
| 实测 | **4,427 一致 / 0 大小不符 / 0 缺失 / 0 多余** |

分片命名 `data/train-00000-of-03388.parquet` 中的 `03388` 与 4,419 不一致，
但**逐文件字节比对全部通过**，说明是上游命名遗留，不是缺片。

### 2.3 X-Atlas-Orion

| 项 | 值 |
|---|---|
| 来源 | `huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion` |
| 路径 | `X-Atlas-Orion/` |
| 期望 | 337 文件 / 126.26 GB（333 parquet + gene_metadata） |
| 实测 | **337 一致 / 0 大小不符 / 0 缺失 / 0 多余** |

### 2.4 Tabula Sapiens v2（cellxgene 合并版）

| 项 | 值 |
|---|---|
| 来源 | CELLxGENE Discover，dataset_version_id `1c88f927-bcbb-4bb1-9881-281842945a2d` |
| 路径 | `Tabula-Sapiens-v2-CELLxGENE/Tabula_Sapiens_v2_All_Cells.h5ad` |
| 大小 | 45,013,733,785 字节（45.01 GB） |
| 细胞 | 1,136,218 |
| 基因 | 60,606（Ensembl 索引 + `feature_name` 符号） |
| 核验 | **重算 sha256 = 记录值** `9ac3d9391bd6211f28454ddd1d7bff0706c286c0cacb909179dd0f9cde68275a` |
| 下载日志 | `START 12:59 → VERIFYING 22:46 → COMPLETE 22:51` |

该文件**不是** figshare 上 `Tabula_Sapiens_v2`（article 27921984）的 28 个按器官文件。
下载器自己的 manifest 写明：

```json
"requested_figshare": "https://figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984",
"note": "Official CELLxGENE-curated all-cells alternative; not the Figshare files."
```

### 2.5 Tabula Sapiens v2（figshare 28 器官文件）

| 项 | 值 |
|---|---|
| 来源 | `figshare.com/articles/dataset/Tabula_Sapiens_v2/27921984` |
| 期望 | 28 文件 / 57.14 GB（按器官拆分：Bladder、Blood、Bone_Marrow、…、Vasculature） |
| 状态 | **未下载** |

与 2.4 的内容重叠（2.4 是同一批细胞的合并版）。对预训练而言合并版更合用，
不需要先合并 28 个器官文件。是否补下取决于是否需要按器官拆分的一份。

---

## 三、基因轴统一性

| 数据源 | 基因标识 | 数量 |
|---|---|---|
| Arc | 基因符号 | 18,080 |
| Tabula Sapiens | Ensembl（`_index`）+ 符号（`feature_name`） | 60,606 / 59,410 符号 |
| Orion | 符号 + Ensembl | 38,584 符号 |
| Tahoe | 符号 + Ensembl | 62,710 符号 |

重叠（按基因符号）：

```
Arc ∩ Tabula   17,635   97.5% of Arc
Arc ∩ Orion    17,701   97.9% of Arc
Arc ∩ Tahoe    17,741   98.1% of Arc
Arc ∩ 任一外部  17,741   98.1% of Arc
```

因此预训练面板取 **4,096 个基因**，全部同时存在于 Arc 与 Tabula。
面板按**外部数据方差**选择，不看 Arc 扰动细胞的表达。

高方差基因前 10（Tabula，跨器官）：`SRGN, PTPRC, ARHGAP15, CXCR4, CD74, S100A9, CST3, VIM, S100A4, SMAP2`
—— 免疫与结构标记，符合多器官图谱的预期。

---

## 四、复现核验

```bash
V=/mnt/storage/bzhang/AIVC/c3-vcell/.venv/bin/python
cd /mnt/storage/bzhang/AIVC/vcellx

# 拉取 HF 完整清单（本地运行，需外网）
python tools/fetch_manifest.py

# 把 remote_manifest.json 传到服务器后逐文件比对
$V tools/audit_local.py remote_manifest.json

# Tabula Sapiens 的 sha256 重算（45 GB，约数分钟）
$V tools/audit_tabula_local.py

# 基因轴重叠
$V tools/audit_gene_axes.py
```

核验输出（2026-11-16）：

```
### arc    expected 5 files, 15.48 GB     identical 5      wrong 0  missing 0  extra 0
### tahoe  expected 4,427 files, 428.81 GB identical 4,427  wrong 0  missing 0  extra 0
### orion  expected 337 files, 126.26 GB   identical 337    wrong 0  missing 0  extra 0
### tabula sha256 MATCH 9ac3d9391bd6...8275a
```

---

## 五、本仓库不做的事

- **不重新分发**任何原始数据或模型权重
- **不使用**任何 STATE 权重
- **不引用**未经本仓库跑出的性能数字；论文数字出现时一律标注来源
