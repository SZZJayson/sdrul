# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing a **Continual Learning Framework for Turbo Engine Remaining Useful Life (RUL) Prediction**. The research addresses three core pain points in aircraft engine health management:

1. **New operating condition adaptation**: Engine sensor data distribution shifts across different routes, seasons, and altitudes
2. **Catastrophic forgetting**: Models lose prediction capability for old conditions when adapting to new ones
3. **Resource constraints**: On-board/edge devices cannot support large-scale retraining

### Research Hypotheses

- **H1**: Diffusion model-based generative replay can produce pseudo-samples preserving degradation dynamics, mitigating catastrophic forgetting
- **H2**: Self-distillation can be extended to regression tasks via "degradation trajectory conditioning" for online adaptation without storing raw data
- **H3**: Mixture-of-Experts with degradation-stage-aware routing can organize multi-condition knowledge more effectively than single models

## Planned Architecture

The framework integrates three main components:

### 1. Degradation-Conditioned Diffusion (DCD) - Replay Module
- Generates pseudo time-series samples preserving degradation dynamics
- Uses trajectory shape (monotonicity, degradation rate, inflection points) as conditioning
- Key metrics: MPR (Monotonicity Preservation Rate), TSS (Trend Similarity Score), PR (Predictive Retention)

### 2. Trajectory-Conditioned Self-Distillation (TCSD) - Adaptation Module
- Replaces in-context learning with "trajectory prototypes" for small models
- Teacher branch: conditioned on trajectory prototypes
- Student branch: learns from generated trajectories via distillation
- Loss: Distributional RUL distillation (Gaussian KL + Wasserstein)

### 3. Degradation-Stage-Aware MoE (DSA-MoE) - Knowledge Organization
- 2D expert matrix: Operating Conditions × Degradation Stages
- Joint routing strategy: condition encoder + health indicator
- Supports both hard routing (known conditions) and soft routing (unknown conditions)

## Research Roadmap

```
Phase 1 (3 months)          Phase 2 (4 months)           Phase 3 (5 months)
    ↓                           ↓                           ↓
┌─────────────┐           ┌─────────────┐           ┌─────────────┐
│ Diffusion   │           │ Self-       │           │ SDFT-DIMIX  │
│ Replay      │    →      │ Distillation│    →      │ Integration │
│ Validation  │           │ for RUL     │           │             │
└─────────────┘           └─────────────┘           └─────────────┘
```

## Datasets

- **C-MAPSS**: NASA turbo engine degradation data (FD001-FD004 sub-datasets)
- **N-CMAPSS**: New C-MAPSS with real flight profiles (Units 2, 5, 10, 14/15)

## Project Structure (Planned)

```
sdrul/
├── data/
│   ├── cmapss/           # C-MAPSS data processing
│   └── ncmapss/          # N-CMAPSS data processing
├── models/
│   ├── dcd/              # Degradation-Conditioned Diffusion
│   ├── tcsd/             # Trajectory-Conditioned Self-Distillation
│   ├── moe/              # Mixture of Experts
│   └── encoders/         # Feature extractors
├── continual/
│   ├── replay.py         # Replay buffer management
│   └── metrics.py        # CL metrics (ACC, BWT, FWT)
├── experiments/
│   ├── phase1/           # DCD validation
│   ├── phase2/           # TCSD validation
│   └── phase3/           # Full framework
└── utils/
    ├── trajectory.py     # Trajectory shape extraction
    └── health_indicator.py # HI computation
```

## Key Implementation Notes

### Degradation Trajectory Encoding
When implementing trajectory shape extraction:
- First-order difference: captures degradation rate
- Second-order difference: captures degradation acceleration
- Normalized shape: removes absolute RUL value effects

### Evaluation Metrics for Continual Learning
- **ACC** (Average Accuracy): Average performance across all tasks
- **BWT** (Backward Transfer): Measures forgetting (negative = forgetting occurred)
- **FWT** (Forward Transfer): Knowledge transfer from old to new tasks

### Training Strategy
Joint training combines three losses:
1. Supervised loss on new data
2. Distillation loss on replay samples
3. Load balancing loss for expert regularization

## Recent Research Context (2024-2025)

### Key Papers to Reference
- **Self-Distillation Enables Continual Learning** (arXiv 2601.19897, Jan 2025): SDFT framework - core inspiration for TCSD
- **Diffusion-TS** (ICLR 2024): Interpretable diffusion for time series generation - baseline for DCD
- **Mixture of Experts Meets Prompt-Based CL** (NeurIPS 2024): MoE routing mechanisms for continual learning
- **SDDGR** (CVPR 2024): Stable Diffusion-based generative replay for continual learning
- **EWC-Guided Diffusion Replay** (2024): Hybrid approach combining EWC with diffusion replay
- **Transformer-based RUL** (Nature Scientific Reports 2024): SOTA RUL prediction with transformers

### Architecture Updates Based on Latest Research
1. **Transformer-based Feature Extractor**: Replace CNN with Transformer following 2024 RUL literature trends
2. **Adapter-based Experts**: Use lightweight Adapter experts (CVPR 2024 MoE work) for efficiency
3. **EWC Regularization**: Add EWC term to joint training loss (proven effective in 2024 papers)
4. **Extended Evaluation Metrics**: Add DSP (Degradation Stage Preservation), DC (Distribution Consistency), UC (Uncertainty Calibration)

## Documentation

- **Architecture Details**: See `docs/architecture.md` for complete system architecture and data flow
- **Research Evaluation**: See `docs/research_framework_evaluation.md` for detailed framework assessment and improvement suggestions

## References to Follow

- **Continual Learning**: NeurIPS/ICML/ICLR continual learning tracks, especially regression and time series work
- **Diffusion Models**: TimeGrad, CSDI, Diffusion-TS (ICLR 2024), DiffRUL
- **RUL Prediction**: IEEE TII, MSSP, RESS journals, Nature Scientific Reports
- **MoE**: NeurIPS 2024 MoE-CL paper, ICLR 2025 MoE theory work
