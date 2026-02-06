# 持续学习框架用于涡轮发动机RUL预测
# Continual Learning Framework for Turbo Engine RUL Prediction

## 项目概述 (Project Overview)

本研究项目开发了一个持续学习框架，用于解决航空发动机剩余使用寿命（RUL）预测中的三个核心挑战：

1. **新工况适应困难** - 发动机在不同航线、季节、高度下运行，传感器数据分布持续变化
2. **灾难性遗忘** - 模型针对新工况调整后，往往丧失旧工况的预测能力
3. **运行时资源受限** - 机载或边缘计算设备无法承载大规模重训练

### 核心创新

- **DCD (Degradation-Conditioned Diffusion)**: 基于退化轨迹条件的扩散模型，生成保持退化动态特性的回放样本
- **TCSD (Trajectory-Conditioned Self-Distillation)**: 轨迹条件自蒸馏，实现无需存储原始数据的在线适应
- **DSA-MoE (Degradation-Stage-Aware MoE)**: 退化阶段感知的混合专家，按工况×阶段二维组织知识

## 研究假设

- **H1**: 扩散模型可以生成保持退化动态特性的伪样本，缓解灾难性遗忘
- **H2**: 自蒸馏可以通过退化轨迹条件化拓展到回归任务
- **H3**: 混合专家架构结合退化阶段感知路由，可以有效组织多工况知识

## 数据集

| 数据集 | 描述 | 场景 |
|--------|------|------|
| **C-MAPSS** | NASA涡扇发动机退化数据 | FD001-FD004 (不同工况/故障模式) |
| **N-CMAPSS** | 新C-MAPSS，真实飞行剖面 | Unit 2, 5, 10, 14/15 |

## 项目结构

```
sdrul/
├── CLAUDE.md                  # Claude Code 项目指南
├── README.md                  # 项目说明 (本文件)
├── docs/
│   ├── architecture.md        # 系统架构详细设计
│   └── research_framework_evaluation.md  # 研究框架评估与改进建议
├── figures/                   # 图表和可视化
└── (待开发代码目录)
```

## 研究路线图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Phase 1       │     │   Phase 2       │     │   Phase 3       │
│   (3个月)       │     │   (4个月)       │     │   (5个月)       │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ DCD验证         │  →  │ TCSD验证        │  →  │ 完整框架评估    │
│ - 扩散回放      │     │ - 自蒸馏        │     │ - 端到端测试    │
│ - 时序连贯性    │     │ - 在线适应      │     │ - 基准对比      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 评估指标

### 持续学习指标
- **ACC** (Average Accuracy): 所有任务平均性能
- **BWT** (Backward Transfer): 后向迁移，衡量遗忘程度
- **FWT** (Forward Transfer): 前向迁移，衡量知识迁移

### 生成质量指标
- **MPR** (Monotonicity Preservation Rate): 退化单调性保持率
- **TSS** (Trend Similarity Score): 趋势相似度 (DTW距离)
- **PR** (Predictive Retention): 预测保持性

### RUL预测指标
- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **Uncertainty Calibration**: 不确定性校准

## 最新研究背景 (2024-2025)

本项目基于以下最新研究进展：

- **Self-Distillation Enables Continual Learning** (arXiv 2025): SDFT框架
- **Diffusion-TS** (ICLR 2024): 时序扩散模型SOTA
- **MoE Meets Prompt-Based CL** (NeurIPS 2024): 专家混合持续学习
- **Transformer-based RUL** (Nature 2024): 基于Transformer的RUL预测

详见 `docs/research_framework_evaluation.md`。

## 开发计划

### 第一阶段：DCD验证
- [ ] 实现退化轨迹条件扩散模型
- [ ] 在C-MAPSS FD001上验证生成质量
- [ ] 评估MPR、TSS、PR指标

### 第二阶段：TCSD验证
- [ ] 实现轨迹原型管理器
- [ ] 实现分布RUL蒸馏损失
- [ ] 验证在线适应效果

### 第三阶段：完整框架
- [ ] 集成DCD、TCSD、DSA-MoE
- [ ] 端到端持续学习实验
- [ ] 与SOTA方法对比

## 许可证

待定

## 联系方式

待定

## 参考文献

详见各阶段实验报告和文档。
