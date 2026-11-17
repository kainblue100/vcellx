# vcellx · 表观先验到底有没有用？一个可审计的对照实验

> 两个虚拟细胞模型共用同一骨干、同一数据、同一训练预算，**只改变表观信号进入模型的方式**：
> 一条让它沿基因调控网络传播，一条直接编码。第三个臂用**度匹配打乱的拓扑**做对照，
> 用来区分「图拓扑有用」和「只是看到了更多基因」。
>
> 本文档按 [ArcInstitute/state](https://github.com/ArcInstitute/state) 的组织方式编写。

**研究状态：原型，结果未完成。** 预训练编码器**已训练完成**（8 epoch，掩码重建损失
0.680 → 0.576，3.41 M 参数，见 `work/foundation/encoder.pt`）；三臂扰动模型与
`cell-eval2` 评测**尚未运行**。**本仓库不包含任何未真实跑出的性能数字**，
结果表格中的空位会保留为 `pending` 直到对应实验完成。这里的代码没有一个是
STATE 的复现，也没有使用任何 STATE 的权重。

---

## 一、要回答的问题

单细胞扰动实验能测出「敲掉某个基因之后细胞变成什么样」，但**表观遗传先验能不能帮助预测这件事，目前没有定论**。
文献里两种说法都有人支持，而多数比较混淆了三件事：

1. 图**拓扑**是否携带信息
2. 模型是否只是**看到了更多基因**（图传播天然引入邻居）
3. 是否**过拟合**了训练集里的靶基因

本仓库的设计目标是让这三件事**可以分开测量**。

---

## 二、模型设计

### 2.1 共享骨干

```
表达编码器  E : R^G -> R^d        (MLP + GELU, G = 4096, d = 256)
扰动注入    h_i = E(x_i) + z_p     (残差式注入)
解码器      D : R^d -> R^G
损失        MMD(预测细胞集, 观测扰动细胞集)
```

`x_i` 是单个对照细胞的表达向量；`z_p` 是靶基因 `p` 的条件向量；输出是同一批细胞在扰动之后的表达。

**为什么用集合而不是单细胞**：单细胞测序是破坏性的，扰动前的 `X_0` 与扰动后的 `X_p`
不可能来自同一个细胞，逐细胞配对在原理上不成立。因此把损失定义在两个**细胞集合的分布**之间，
用最大均值差异（MMD，能量距离核）比较。这与 STATE 的动机一致，但实现是独立的。

### 2.2 三个臂，只改一个变量

设 `e_g ∈ R^k` 为基因 `g` 的表观特征向量，`A` 为基因调控网络的邻接矩阵（行归一化）。

| 臂 | 条件向量 | 它检验什么 |
|---|---|---|
| **graph** | `z_p = emb[p] + W · (A e)_p` | 表观信号沿 GRN 扩散 |
| **shuffle** | `z_p = emb[p] + W · (A_shuf e)_p` | **同度分布、不同拓扑** |
| **direct** | `z_p = emb[p] + W · e_p` | 只用靶基因自身特征，无图 |

三臂的 `emb`（学出来的逐靶基因嵌入）、编码器、解码器、优化器、步数、数据**完全相同**。

这个三分法是整个仓库的核心。没有 `shuffle` 臂，`graph > direct` 无法解释 ——
图传播让模型看到了邻居基因的表观特征，赢了也可能只是信息量更大。`shuffle` 保持每个节点的度不变
（通过受约束的换边），因此「看到多少基因」被固定住，**唯一变化的是谁连谁**。

### 2.3 GRN 的构造规则

**第一版方案已实测否决**（细节见 [`docs/method.md`](docs/method.md#3-grn-构造)）：
用 K562 的 Hi-C 环调用（全基因组 52,276 个环）在 2000 基因面板上只得到 **15 条边**，
中位度数 0；把启动子窗口放宽到 ±100 kb 后中位度数仍是 0。

采用的方案是 **bin 级接触矩阵**（4DN HFFc6 Hi-C，`.mcool` 直接用 h5py 读取）：
基因 TSS → 100 kb bin，用 `bin1_offset` 索引只读含面板基因的 bin，
保留计数在 90 分位以上的强接触，每节点 top-10。

```
边数        12,015
孤立基因    50 / 2000
中位度数    11      平均 12.0    最大 28
```

**解释边界**：100 kb 的一个 bin 里通常有好几个基因，所以一条边的含义是
「这两个基因处在互相接触的 100 kb 区间内」，**不是**「两个启动子之间有环」。
这条说明写进图摘要并随结果一起报告。

建图过程**不看表达数据**，也不看留出靶基因的响应。启动子坐标来自 GENCODE v44。

### 2.4 预训练基础模型

在扰动模型之前，先在一个跨数据集的细胞语料上训练一个细胞编码器：

```
输入    4096 基因面板的 log1p-CP10k 表达
掩码    每个细胞随机遮住 30% 的基因
编码    剩余 70% → 单个 256 维向量
解码    只从该向量重建被遮住的位置
```

**只重建被遮住的位置**是关键：否则编码器会退化成恒等映射。跨数据集共享同一基因面板，
这一步才算「预训练表示」而不是「每个数据集一个自编码器」。

面板选择只用**外部数据**的方差（不看 Arc 扰动细胞），且要求基因同时存在于 Arc 与外部语料。

---

## 三、数据

### 3.1 来源与完整性

所有数据位于 `10.13.8.200:/mnt/storage/bzhang/AIVC`。完整性与逐文件核验见
[`docs/provenance.md`](docs/provenance.md)。

| 数据集 | 规模 | 核验方式 | 状态 |
|---|---|---|---|
| Arc Virtual Cell Challenge 训练集 | 221,273 细胞 / 18,080 基因 / 150 靶 + 38,176 对照 | HF 逐文件字节比对 5/5 | ✅ |
| Tahoe-100M | 4,419 parquet 分片 | HF 逐文件字节比对 4,427/4,427 | ✅ |
| X-Atlas-Orion | 333 parquet 分片 | HF 逐文件字节比对 337/337 | ✅ |
| Tabula Sapiens v2（cellxgene） | 1,136,218 细胞 / 60,606 基因 | **sha256 重算与记录一致** | ✅ |
| Tabula Sapiens v2（figshare 28 器官文件） | 57.14 GB | — | ⬜ 未下载 |

### 3.2 基因轴统一

```
Arc ∩ Tabula 17,635 (97.5%)   Arc ∩ Orion 17,701 (97.9%)   Arc ∩ Tahoe 17,741 (98.1%)
```

预训练面板取 4,096 个基因，全部同时存在于 Arc 与 Tabula。

### 3.3 切分

**靶基因不相交**：150 个靶基因切成 98 训练 / 22 验证 / 30 测试。
同一个靶基因不会同时出现在训练和测试里 —— 同靶切分是这类模型虚高指标的常见来源。

HVG 选择与归一化统计**只在对照细胞上做**。

---

## 四、快速开始

```bash
# 环境复用服务器上已有的解释器，不新建 venv（该机 python3 缺 ensurepip）
V=/mnt/storage/bzhang/AIVC/c3-vcell/.venv/bin/python
export PYTHONPATH=/mnt/storage/bzhang/AIVC/vcellx/src

# 1 载入 Arc、归一化、选 HVG、靶基因不相交切分
$V -m vcellx.run_all --stage data --work work

# 2 下载 GENCODE + 用 K562 peak 构建每基因表观特征
$V -m vcellx.run_all --stage epi --work work

# 3 用 K562 Hi-C 环构建 GRN + 度匹配打乱对照
$V -m vcellx.run_all --stage graph --work work

# 4 预训练面板 → 缓存 Tabula → 训练编码器
$V -m vcellx.run_pretrain --step panel --work work --panel 4096
$V -m vcellx.run_pretrain --step cache --work work --panel 4096 --cache-cells 400000
$V -m vcellx.run_pretrain --step train --work work --panel 4096 \
   --train-cells 150000 --epochs 8 --hidden 512
```

---

## 五、模块

| 文件 | 职责 |
|---|---|
| `src/vcellx/data.py` | 载入 / 归一化 / HVG（仅对照）/ 靶基因不相交切分 / 集合打包 |
| `src/vcellx/epi.py` | GENCODE TSS + K562 ATAC 与 TF ChIP → 启动子表观特征 |
| `src/vcellx/grn.py` | K562 Hi-C 环 → GRN，以及度匹配拓扑打乱 |
| `src/vcellx/pretrain.py` | 掩码表达自编码器（基础模型） |
| `src/vcellx/model.py` | 共享骨干 + 三臂条件化 + MMD 集合损失 |
| `src/vcellx/run_all.py` | data / epi / graph 阶段驱动 |
| `src/vcellx/run_pretrain.py` | 预训练阶段驱动 |
| `tools/` | 数据完整性与基因轴核验脚本 |

---

## 六、评测计划

用 `cell-eval2 0.22.0` 的官方 `vcc2026` preset 打分，覆盖三类读出口：

1. **表达量** —— 扰动后表达变化的 Pearson 相关
2. **差异表达** —— DE 基因的 AUPRC、倍数变化 Spearman、top-k 重合率
3. **效应量** —— 扰动的整体强度排序

对照基线：

| 基线 | 说明 |
|---|---|
| control-only | 直接复制输入对照细胞，不建模任何响应 |
| mean-response | 用训练集所有扰动的平均响应，与靶基因无关 |
| `direct` 臂 | 本仓库的无图条件化 |
| `shuffle` 臂 | 同度不同拓扑 |
| STATE（论文数字） | **标注为论文公布值，不是本仓库复现** |

### 结果

| 指标 | control | mean | direct | shuffle | graph | STATE（论文） |
|---|---|---|---|---|---|---|
| 表达变化 Pearson | pending | pending | pending | pending | pending | 不适用 |
| DE AUPRC | pending | pending | pending | pending | pending | 不适用 |
| 效应量 Spearman | pending | pending | pending | pending | pending | 不适用 |

**这张表在实验跑完之前不会填任何数字。** 论文数字与我方数字的数据集、切分、上下文都不同，
直接并排比较本身就有误导性；表格存在的意义是让读者看到「同一协议下的四个数」，
而不是制造一个跨协议排行榜。

---

## 七、已知限制（重要）

### 7.1 表观先验与表达数据不是同一个细胞背景

这是本设计最大的方法学弱点，必须写在显眼处：

- Arc VCC 训练集的 README **没有写明细胞系**
- 磁盘上可用的公共表观数据是 **K562**（ATAC、CTCF/SP1/USF1 ChIP、Hi-C、WGBS）
- 因此本仓库的 GRN 是**跨背景的人类调控先验**，不是为该表达数据匹配的网络

这会让「图是否有用」的检验更难跑出正结果，也意味着即使 `graph` 胜出，
也不能直接推断「在这个细胞里表观调控是这样的」。所有相关结论都必须带上这一条。

### 7.2 其它

- 预训练语料目前只用了 Tabula Sapiens；Orion 与 Tahoe 的观测细胞尚未加入
- 图传播是单跳的（`A e`），没有做多跳或注意力加权
- WGBS 甲基化特征尚未接入，当前特征只有 ATAC 与三个 TF
- 集合大小固定 64，未做超参搜索
- 服务器为共享环境，实测只有约 4 核可用（请求 16 线程），预训练约 17 分钟/epoch

---

## 八、引用与许可

- CODA 无关。表观数据来自 ENCODE（K562 系）。
- Arc VCC 数据集、Taboe-100M、X-Atlas-Orion、Tabula Sapiens 各自遵循其原始许可，
  本仓库不重新分发任何原始数据或权重。
- 本仓库代码为研究原型，未经过同行评议。
