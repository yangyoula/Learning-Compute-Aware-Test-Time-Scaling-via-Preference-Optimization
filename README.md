 Learning Compute-Aware Test-Time Scaling via Preference Optimization (POTTS)

This repository provides a minimal implementation of **Preference-Optimized Test-Time Scaling (POTTS)**, a framework for adaptive inference in vision models.

## 📌 Overview

Test-time scaling improves prediction performance by allocating additional computation during inference. However, fixed compute budgets are inefficient, as different inputs require varying levels of computational effort.

POTTS addresses this problem by learning a **compute-aware policy** that dynamically selects inference strategies (e.g., single-pass or test-time augmentation) based on input features and uncertainty signals.

## 🚀 Key Idea

- Construct **pairwise preferences** between inference strategies  
- Learn a policy to balance:
  - prediction accuracy  
  - computational cost  
- Perform **adaptive inference** conditioned on input difficulty  

## 📊 Supported Strategies

- Single-pass inference (K=1)
- Test-time augmentation (K=4)
- Test-time augmentation (K=8)

## 📁 Repository Structure


.
├── notebook/ # Colab / experiment notebook
├── models/ # Policy network (MLP)
├── data/ # Cached features (optional)
├── results/ # Output metrics and logs
├── acc_vs_compute.png # Accuracy vs compute figure
├── stress_vs_error.png # Uncertainty vs error analysis
├── policy_compute_hist.png # Compute allocation visualization
└── README.md


## ⚙️ Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## 🧪 Running Experiments

You can reproduce the main results using the provided notebook:

```bash
jupyter notebook Untitled16.ipynb

or open in Google Colab.

📈 Results

Key observations:

Uncertainty (entropy) correlates strongly with prediction errors
Threshold-based heuristics are strong baselines
POTTS achieves:
competitive accuracy
improved calibration (lower ECE)
Adaptive compute allocation reflects input difficulty
📄 Paper

This code accompanies the paper:

"Learning Compute-Aware Test-Time Scaling via Preference Optimization"

🔓 Reproducibility

All experiments can be run on a single GPU within a short time (≈1 hour on A100).

⚠️ Note

This is a simplified research implementation intended for clarity and reproducibility rather than production use.

📬 Contact

Youla Yang
Indiana University Bloomington
Email: yangyoul@iu.edu
