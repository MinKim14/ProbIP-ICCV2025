# ProbIP: Probabilistic Inertial Poser

**📰 [NEW] Accepted at ICCV 2025!**  
**🚀 Code and models coming soon — stay tuned!**

---

## 🌀 Overview

**ProbIP (Probabilistic Inertial Poser)** is a novel probabilistic framework for human motion estimation using sparse wearable IMU sensors. Unlike existing methods that rely on physical priors or heuristics, ProbIP directly models uncertainty in motion prediction through probabilistic representations — enabling robust and accurate full-body motion reconstruction, even under sparse sensor configurations.

---

## 🔬 Key Features

- **RU-Mamba Blocks**: We introduce Rotation-Uncertainty Mamba (RU-Mamba), a new sequence model block that outputs a **matrix Fisher distribution** over rotation matrices, capturing both motion and its uncertainty.
  
- **PDN (Progressive Distribution Narrowing)**: A novel mechanism that gradually refines the motion distribution across layers to stabilize training and improve prediction quality across diverse motion types.

- **Sensor Efficient**: ProbIP achieves **state-of-the-art results with just six IMUs** and remains competitive even with fewer sensors, making it ideal for real-world applications with hardware constraints.

---

## 📊 Experimental Highlights

- Achieves **SOTA accuracy** on multiple public datasets with only 6 IMUs.
- Robust to sensor dropout and reduced sensor scenarios.
- Outperforms existing physically-constrained models in both accuracy and uncertainty modeling.

---

## 📁 Coming Soon

We are finalizing the codebase and pretrained models for public release. The repository will include:

- Training and inference code
- Preprocessing scripts for popular IMU datasets
- Demo notebooks
- Pretrained model checkpoints

---

## 📝 Citation

If you find our work useful, please cite:
