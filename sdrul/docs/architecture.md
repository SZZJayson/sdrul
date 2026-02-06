# 研究架构设计文档

## 系统架构概述

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    持续学习框架用于涡轮发动机RUL预测                                    │
│              Continual Learning Framework for Turbo Engine RUL Prediction              │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                  INPUT LAYER                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    Multi-Sensor Time Series Input                           │    │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐                 │    │
│  │  │Temperature │ │  Pressure  │ │  Vibration │ │ Fan Speed  │                 │    │
│  │  │   (°C)     │ │   (psi)    │ │   (g)      │ │   (rpm)    │                 │    │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘                 │    │
│  │                              Shape: [batch, seq_len, sensor_dim]             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FEATURE EXTRACTION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │              Transformer-based Feature Extractor                              │    │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Sensor Embedding: Linear(sensor_dim → d_model=256)                  │   │    │
│  │  ├──────────────────────────────────────────────────────────────────────┤   │    │
│  │  │  Health-Aware Positional Encoding:                                    │   │    │
│  │  │    - Temporal position encoding                                       │   │    │
│  │  │    - Health stage condition injection                                 │   │    │
│  │  ├──────────────────────────────────────────────────────────────────────┤   │    │
│  │  │  Multi-Head Self-Attention (n_head=8, num_layers=4)                   │   │    │
│  │  │    - Captures long-range temporal dependencies                        │   │    │
│  │  │    - Multi-scale feature extraction                                  │   │    │
│  │  ├──────────────────────────────────────────────────────────────────────┤   │    │
│  │  │  Multi-Scale Temporal Convolution:                                    │   │    │
│  │  │    - Kernel sizes: [3, 5, 7, 9]                                       │   │    │
│  │  │    - Captures local degradation patterns                              │   │    │
│  │  └──────────────────────────────────────────────────────────────────────┘   │    │
│  │                                                                              │    │
│  │  Output: Feature Representation [batch, seq_len, d_model=256]               │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           DEGRADATION-STAGE-AWARE MoE LAYER                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    Gating Network (Router)                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Condition Router: Operating condition classification                 │   │    │
│  │  │    - Input: Feature representation                                   │   │    │
│  │  │    - Output: P(condition | features) ∈ R^(num_conditions)            │   │    │
│  │  ├─────────────────────────────────────────────────────────────────────┤   │    │
│  │  │  Stage Router: Health stage estimation                               │   │    │
│  │  │    - Input: Computed health indicator                                │   │    │
│  │  │    - Output: P(stage | HI) ∈ R^(num_stages=3)                       │   │    │
│  │  └─────────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                        ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │              Expert Matrix (Condition × Stage)                              │    │
│  │  ┌───────────────┬───────────────┬───────────────┐                         │    │
│  │  │               │  Early Stage  │ Middle Stage  │  Late Stage    │         │    │
│  │  ├───────────────┼───────────────┼───────────────┼───────────────┐         │    │
│  │  │ Condition A   │  Expert_A1    │  Expert_A2    │  Expert_A3    │         │    │
│  │  ├───────────────┼───────────────┼───────────────┼───────────────┤         │    │
│  │  │ Condition B   │  Expert_B1    │  Expert_B2    │  Expert_B3    │         │    │
│  │  ├───────────────┼───────────────┼───────────────┼───────────────┤         │    │
│  │  │ Condition C   │  Expert_C1    │  Expert_C2    │  Expert_C3    │         │    │
│  │  │     ...       │     ...       │     ...       │     ...       │         │    │
│  │  └───────────────┴───────────────┴───────────────┴───────────────┘         │    │
│  │                                                                              │    │
│  │  Each Expert: AdapterExpert(d_model=256, bottleneck=64)                      │    │
│  │    - Down projection: d_model → bottleneck                                  │    │
│  │    - GELU activation                                                         │    │
│  │    - Up projection: bottleneck → d_model                                    │    │
│  │    - Residual connection with learned scale                                 │    │
│  │                                                                              │    │
│  │  Output: Weighted combination of expert outputs                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT LAYER                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         RUL Prediction Head                                  │    │
│  │  ┌──────────────────────────────────────────────────────────────────────┐   │    │
│  │  │  Global Average Pooling → MLP Head                                   │   │    │
│  │  │                                                                      │   │    │
│  │  │  Output 1: Point Estimate                                            │   │    │
│  │  │    RUL_pred = μ ∈ R+ (remaining cycles)                              │   │    │
│  │  │                                                                      │   │    │
│  │  │  Output 2: Uncertainty Quantification                                │   │    │
│  │  │    σ = f_uncertainty(features) ∈ R+                                  │   │    │
│  │  │    Confidence Interval: [μ - 1.96σ, μ + 1.96σ]                       │   │    │
│  │  └──────────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════

                            CONTINUAL LEARNING MODULES

┌──────────────────────────────────────────────┐     ┌──────────────────────────────────────────────┐
│           DCD MODULE                          │     │           TCSD MODULE                          │
│  (Degradation-Conditioned Diffusion)          │     │  (Trajectory-Conditioned Self-Distillation)  │
├──────────────────────────────────────────────┤     ├──────────────────────────────────────────────┤
│                                              │     │                                              │
│  Purpose: Generate pseudo-samples            │     │  Purpose: Online adaptation without          │
│           preserving degradation dynamics    │     │           storing raw data                    │
│                                              │     │                                              │
│  ┌────────────────────────────────────────┐ │     │  ┌────────────────────────────────────────┐ │
│  │  Trajectory Shape Encoder              │ │     │  │  Trajectory Prototype Manager          │ │
│  │  - First-order difference (rate)       │ │     │  │  - K-means clustering in shape space   │ │
│  │  - Second-order difference (accel)    │ │     │  │  - Online prototype update (EMA)       │ │
│  │  - Normalized degradation shape        │ │     │  │  - Novelty detection for new patterns  │ │
│  └────────────────────────────────────────┘ │     │  └────────────────────────────────────────┘ │
│                  ↓                           │     │                  ↓                           │
│  ┌────────────────────────────────────────┐ │     │  ┌────────────────────────────────────────┐ │
│  │  Conditional Denoising U-Net           │ │     │  │  Teacher Branch (prototype-cond.)      │ │
│  │  - Time-conditioned diffusion steps    │ │     │  │  ┌──────────────────────────────────┐ │ │
│  │  - Trajectory shape injection          │ │     │  │  │ Current input + Trajectory       │ │ │
│  │  - Condition: c = [shape, condition]   │ │     │  │  │ Prototype → Feature Extractor     │ │ │
│  └────────────────────────────────────────┘ │     │  │  │       → RUL Prediction (μ_T,σ_T) │ │ │
│                  ↓                           │     │  │  └──────────────────────────────────┘ │ │
│  ┌────────────────────────────────────────┐ │     │  └────────────────────────────────────────┘ │
│  │  Replay Generation                     │ │     │                  ↓ distillation          │
│  │  For each condition c:                 │ │     │  ┌────────────────────────────────────────┐ │
│  │    1. Sample prototype p ~ P_c         │ │     │  │  Student Branch (unconditional)        │ │
│  │    2. Generate x_gen ~ Diffusion(p)    │ │     │  │  ┌──────────────────────────────────┐ │ │
│  │    3. Store (x_gen, c) for replay      │ │     │  │  │ Generated trajectory → Feature   │ │ │
│  └────────────────────────────────────────┘ │     │  │  │ Extractor → RUL (μ_S,σ_S)       │ │ │
│                                              │     │  │  └──────────────────────────────────┘ │ │
│  Evaluation Metrics:                          │     │  └────────────────────────────────────────┘ │
│  • MPR (Monotonicity Preservation Rate)      │     │                                              │
│  • TSS (Trend Similarity Score)              │     │  Loss: Distributional RUL Distillation     │
│  • PR (Predictive Retention)                 │     │  L_distill = α·KL + (1-α)·W²                │
└──────────────────────────────────────────────┘     └──────────────────────────────────────────────┘
           │                                                        │
           │ Replay samples                                        │ Soft labels
           └────────────────────────┬───────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              JOINT TRAINING LOOP                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                     Training Components                                      │    │
│  │                                                                              │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │    │
│  │  │   New Data  │    │   Replay    │    │   Self-     │    │    EWC      │   │    │
│  │  │             │    │   Samples   │    │ Distillation│    │ Regularizer │   │    │
│  │  │ (x_new, y)  │    │ (x_rep, y)  │    │ (μ_T,σ_T)   │    │  (Fisher)   │   │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │    │
│  │        │                  │                  │                  │           │    │
│  │        ↓                  ↓                  ↓                  ↓           │    │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │    │
│  │  │                          Loss Functions                             │  │    │
│  │  │                                                                    │  │    │
│  │  │  L_new = MSE(y_pred, y_true)                                       │  │    │
│  │  │  L_replay = Σ KL(N(μ_T,σ_T) || N(μ_S,σ_S))                         │  │    │
│  │  │  L_balance = Σ (load_i - target_load)²                             │  │    │
│  │  │  L_ewc = Σ F_j · (θ_j - θ_old_j)²                                  │  │    │
│  │  │                                                                    │  │    │
│  │  │  L_total = L_new + λ₁·L_replay + λ₂·L_balance + λ₃·L_ewc           │  │    │
│  │  └──────────────────────────────────────────────────────────────────────┘  │    │
│  │                                     ↓                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────────────┐  │    │
│  │  │                    Model Update                                      │  │    │
│  │  │                                                                    │  │    │
│  │  │  1. Update Feature Extractor (all conditions shared)                │  │    │
│  │  │  2. Update Gating Network (condition + stage routing)               │  │    │
│  │  │  3. Update relevant Experts (based on routing)                      │  │    │
│  │  │  4. Update Diffusion Model (new trajectory shapes)                  │  │    │
│  │  │  5. Update Trajectory Prototypes (online EMA)                       │  │    │
│  │  │                                                                    │  │    │
│  │  └──────────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                               │
│                                    └──→ Loops back to Expert Matrix                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════

## 数据流详解

### 前向推理流程 (Inference)
```
1. 输入传感器序列 x ∈ R^(batch × seq_len × sensor_dim)
   ↓
2. 特征提取 f = FeatureExtractor(x) ∈ R^(batch × seq_len × d_model)
   ↓
3. 健康指标计算 hi = HINetwork(f) ∈ R^batch
   ↓
4. 路由权重计算:
   - w_cond = ConditionRouter(f) ∈ R^(batch × num_conditions)
   - w_stage = StageRouter(hi) ∈ R^(batch × num_stages)
   ↓
5. 专家激活与聚合:
   output = Σ_i Σ_j w_cond[:,i] · w_stage[:,j] · Expert_ij(f)
   ↓
6. RUL预测:
   - μ_pred = MLP_head(output)  # 点估计
   - σ_pred = Uncertainty_head(output)  # 不确定性
```

### 持续学习流程 (Continual Learning)
```
当新工况数据到达时:

1. 检测新颖性:
   if novelty_score(new_data, existing_prototypes) > threshold:
       add_new_condition_experts()

2. 更新轨迹原型:
   prototypes = online_update(prototypes, new_trajectories)

3. 生成回放样本:
   for each old_condition c:
       replay_c = DiffusionGenerate(prototype_c, num_samples)

4. 联合训练:
   - 教师分支: 在 replay_c 上使用原型条件预测
   - 学生分支: 在 new_data 上无条件预测
   - 联合优化: L_total = L_new + λ·L_replay + ...

5. 模型更新:
   - 冻结: 旧专家的部分参数 (通过EWC识别重要参数)
   - 更新: 新专家、路由网络、共享特征提取器
```

## 模块间接口设计

### DCD → 主模型接口
```python
class DCDInterface:
    def store_trajectory_shape(self, condition_id, trajectory):
        """存储退化轨迹形状"""
        shape_encoding = self.encode_shape(trajectory)
        self.prototype_buffer[condition_id].append(shape_encoding)

    def generate_replay(self, condition_id, num_samples):
        """生成回放样本"""
        prototype = self.get_prototype(condition_id)
        return self.diffusion_model.sample(prototype, num_samples)
```

### TCSD → 主模型接口
```python
class TCSDInterface:
    def get_teacher_prediction(self, x, prototype):
        """教师预测（原型条件化）"""
        return self.teacher_model(x, condition=prototype)

    def get_student_prediction(self, x):
        """学生预测（无条件）"""
        return self.student_model(x)

    def distillation_loss(self, teacher_out, student_out):
        """蒸馏损失"""
        mu_T, sigma_T = teacher_out
        mu_S, sigma_S = student_out
        return distributional_rul_loss(mu_T, sigma_T, mu_S, sigma_S)
```

### DSA-MoE → 主模型接口
```python
class DSA_MoEInterface:
    def route(self, features, health_indicator, condition_id=None):
        """计算路由权重"""
        if condition_id is not None:
            w_cond = one_hot(condition_id, self.num_conditions)
        else:
            w_cond = self.condition_router(features)
        w_stage = self.stage_router(health_indicator)
        return w_cond, w_stage

    def forward(self, features, w_cond, w_stage):
        """加权专家输出"""
        output = 0
        for i in range(self.num_conditions):
            for j in range(self.num_stages):
                expert_out = self.experts[i][j](features)
                output += w_cond[:, i:i+1] * w_stage[:, j:j+1] * expert_out
        return output
```

## 超参数配置

| 模块 | 超参数 | 默认值 | 说明 |
|------|--------|--------|------|
| Feature Extractor | d_model | 256 | 特征维度 |
| | n_head | 8 | 注意力头数 |
| | num_layers | 4 | Transformer层数 |
| | kernel_sizes | [3,5,7,9] | 多尺度卷积核 |
| DSA-MoE | num_conditions | 6 | 工况数量 |
| | num_stages | 3 | 退化阶段数 |
| | bottleneck_dim | 64 | Adapter瓶颈维度 |
| | temperature | 1.0 | 路由softmax温度 |
| DCD | num_diffusion_steps | 1000 | 扩散步数 |
| | beta_schedule | linear | 噪声调度 |
| | condition_dim | 128 | 条件编码维度 |
| TCSD | num_prototypes | 5 | 每工况原型数 |
| | ema_decay | 0.9 | 原型更新衰减 |
| | alpha_kl | 0.5 | KL散度权重 |
| Training | lambda_replay | 1.0 | 回放损失权重 |
| | lambda_balance | 0.01 | 负载均衡权重 |
| | lambda_ewc | 1000 | EWC正则权重 |
| | learning_rate | 1e-4 | 初始学习率 |
