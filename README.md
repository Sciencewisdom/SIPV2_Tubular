# SIP-v2: Gradient-Anchored Anisotropic Propagation for Thin Tubular Structure Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **核心发现**: Directional propagation priors are domain-invariant mechanisms for topology-aware optimization, whereas topology losses are domain-dependent objectives whose effectiveness depends on structural geometry.

---

## 概览

SIP-v2 (Structure-Induced Propagation v2) 是一个梯度锚定的各向异性传播框架，用于细长管状结构分割（如视网膜血管、道路网络）。与标准CNN的各向同性卷积不同，SIP-v2 通过结构张量将扩散方向锚定到图像梯度，使信息沿结构方向传播而非均匀扩散。

**关键创新**:
- **方向锚定**: 扩散方向来自图像梯度（结构张量），而非自由学习
- **强度调制**: 仅扩散强度 $\lambda_\parallel$ / $\lambda_\perp$ 按位置学习
- **残差修正**: 扩散作为CNN分支的小残差修正

---

## 核心结果

### clDice 不对称效应（本文核心发现）

当拓扑压力（clDice权重 $\lambda_c$）增强时：

| 指标 | CNN (DW) | SIP-v2 | 差异 |
|------|----------|--------|------|
| SkelRec @ λ=0.1 | 0.758 (-0.030) | 0.782 (+0.019) | **相反趋势** |
| Breaks @ λ=0.1 | 436 (+258) | 425 (-28) | **相反趋势** |
| Dice @ λ=0.1 | 0.798 (±0.001) | 0.800 (±0.001) | **稳定** |

**结论**: 同一拓扑损失，在不同架构上产生相反的优化轨迹。这是损失-架构交互的签名。

### 跨数据集性能（视网膜）

| 数据集 | E1 (DW) | E4 (SIP-v2 Min) | **E5 (SIP-v2 Full)** |
|--------|---------|-----------------|----------------------|
| DRIVE | 0.7986 | 0.8007 | **0.8032** |
| CHASE_DB1 | 0.7550 | 0.7596 | **0.7629±0.003** |
| HRF | 0.7379 | 0.7285 | **0.7573±0.002** |

### 跨域验证（道路网络）

| 实验 | 种子 | Dice | APLS | GapRec |
|------|------|------|------|--------|
| R0 (DW) | 42/0/1 | 0.9767±0.003 | 0.7311±0.062 | 0.9851±0.004 |
| R1 (SIP-v2 Road) | 42/0/1 | 0.9740±0.008 | 0.6883±0.081 | 0.9844±0.006 |

**关键发现**: 拓扑损失（clDice）在视网膜血管上协同SIP-v2，但在道路网络上损害图连续性（APLS -0.041, p<0.05）。传播先验是domain-invariant的；拓扑损失是domain-dependent的。

---

## 快速开始

### 环境要求

```bash
Python >= 3.10
PyTorch >= 2.6 (with CUDA)
```

### 安装

```bash
git clone <repo-url>
cd SIPV2_Tubular
pip install -r requirements.txt
```

### 数据准备

将数据集解压到 `data/raw/` 目录：
```
data/raw/
  DRIVE/
    training/
    test/
  CHASE_DB1/
  HRF/
  mass_roads/          # 道路数据集 (可选，跨域验证)
    train/sat/
    train/map/
    valid/sat/
    valid/map/
```

### 训练

```bash
# 训练 SIP-v2 (E4) on DRIVE
python scripts/train.py --exp E4 --dataset DRIVE --seed 42

# 训练 SIP-v2 Full (E5)
python scripts/train.py --exp E5 --dataset DRIVE --seed 42

# 使用 clDice 损失
python scripts/train.py --exp E4 --dataset DRIVE --use_cldice --cldice_lambda 0.3

# 训练道路提取 (跨域验证)
python scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42
```

### 评估

```bash
# 视网膜实验评估
# 评估在训练过程中自动进行，或通过 TensorBoard 查看验证指标

# 道路实验确定性评估
python scripts/eval_road_deterministic.py --checkpoint outputs/road_sipv2_50ep/.../checkpoint_final.pth --block_type sipv2_road
```

---

## 项目结构

```
SIPV2_Tubular/
├── sipv2/              # 核心代码
│   ├── ops/            # 算子 (Sobel, Structure Tensor, Diffusion, NormClip)
│   ├── models/         # 模型 (U-Net, 5种Block变体)
│   ├── datasets/       # 数据加载器 (DRIVE, CHASE_DB1, HRF, Mass Roads)
│   ├── losses/         # 损失函数 (BCE+Dice, clDice)
│   ├── metrics/        # 评估指标 (Dice, clDice, Skeleton, Road Topology)
│   ├── engine/         # 训练/验证引擎
│   └── utils/          # 工具函数
├── scripts/            # 训练/评估脚本
│   ├── train.py              # 视网膜实验
│   ├── train_road.py         # 道路跨域实验
│   ├── eval_road_deterministic.py  # 确定性评估
│   └── ...
├── paper/              # 论文 LaTeX 源文件
├── paper_figures/      # 论文图表
├── outputs/            # 实验结果 + PDF
└── tests/              # 单元测试
```

---

## 实验列表

| 实验 | 描述 | 状态 |
|------|------|------|
| E0 | U-Net (ConvBlock) | ✅ |
| E1 | Depthwise Separable CNN | ✅ |
| E2 | Isotropic Diffusion | ✅ |
| E3 | OldSIP (自由学习张量) | ✅ |
| E4 | SIP-v2 Min (梯度锚定) | ✅ |
| E5 | SIP-v2 Full (+Prototype + Global) | ✅ |
| E1+clDice | CNN + 拓扑损失 | ✅ |
| E4+clDice | SIP-v2 + 拓扑损失 | ✅ |
| P0-1 | clDice λ 扫描 (0.0/0.1/0.3/0.5) | ✅ |
| Multi-seed | 3种子视网膜稳定性验证 | ✅ |
| R0/R1/R2 | 道路跨域验证 (3 seeds) | ✅ |

---

## 论文

- **主论文 v44**: `paper/main.pdf` (23页)
- **补充材料 v44**: `paper/supplementary_main.pdf` (6页)
- **投稿信**: `paper/cover_letter.pdf` (2页, TMI版)
- **提交包**: `submission_package_v44.zip` (14MB, 含全部源文件与图表)

### 核心图表

| 图号 | 内容 | 位置 |
|------|------|------|
| Fig 1 | Hero Mechanism Pipeline | Introduction |
| Fig 2 | Architecture Schematic | Method |
| Fig 3 | clDice Response Curve | Results Q3 (核心) |
| Fig 4 | Ablation Study | Results Q1-Q2 |
| Fig 5 | Cross-Dataset E5 Comparison | Results Q4 |
| Fig 6 | Multi-seed Stability | Results Q5 |
| Fig 7 | Road Metric Decoupling | Results Q6 (跨域) |
| Fig 8 | Road Multi-seed Stability | Results Q6 (跨域) |
| Fig 9 | Failure Case Taxonomy | Discussion |
| Fig 10 | Complexity Benchmark | Results |

---

## 引用

```bibtex
@article{sipv2,
  title={SIP-v2: Gradient-Anchored Anisotropic Propagation for Thin Tubular Structure Segmentation},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

---

## 许可

MIT License
