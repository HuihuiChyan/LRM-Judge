# LRM-as-a-Judge

This repository contains the official implementation and datasets for our **ACL 2026** submission:
> **"Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases"**

---

## 📂 Repository Structure

```text
.
├── datasets/                 # Evaluation benchmarks and raw data
├── results/                  # Execution outputs and evaluation metrics
├── scripts/
│   └── self_synthesized/     # Core logic for PlanJudge
│       ├── evaluate/         # Scoring logic and bias analysis scripts
│       ├── run_self_synthesized_key.py
│       ├── run_self_synthesized_key.sh   # Entry: Heuristic-based planning
│       ├── run_self_synthesized_plan.py
│       └── run_self_synthesized_plan.sh  # Entry: Self-synthesized & Combined planning
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```

## 🚀 Running PlanJudge

We provide two primary execution modes for the **PlanJudge** framework:

#### **A. Heuristic-based Planning**
To perform PlanJudge with heuristic-based planning, run the following script:
```bash
bash scripts/self_synthesized/run_self_synthesized_key.sh
```

#### **B. Self-synthesized & Combined Planning**
To perform PlanJudge with self-synthesized or combined planning, run:
```bash
bash scripts/self_synthesized/run_self_synthesized_plan.sh
```