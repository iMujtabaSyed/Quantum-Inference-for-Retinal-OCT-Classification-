# Delegated Quantum Inference for Retinal OCT over a Noisy Quantum Link

Reproducibility code for the paper *"Accuracy Is Not Enough: Reliability and
Explainability of Delegated Quantum Inference for Retinal OCT Classification
over a Noisy Quantum Link."*

A hybrid quantum transfer-learning classifier (frozen ResNet-18 → PCA → a
variational quantum circuit head) is evaluated as a **networked component**:
the variational head is treated as a *delegated* computation executed on a
remote QPU, and the quality of the noisy quantum link is the independent
variable. The study's focus is **reliability and calibration**, not a quantum
accuracy advantage.

## Key findings
- The quantum head reaches ~0.87 accuracy and is **not competitive** with simple
  classical models on the same 6-D features (logistic regression / SVM ≈ 0.99);
  no quantum advantage is claimed.
- Channel types degrade the workload differently: **depolarizing dominates**,
  amplitude damping is intermediate, **phase damping is least damaging**.
- **Probabilistic reliability degrades before accuracy.** At depolarizing noise
  p=0.10 accuracy is unchanged from clean while Brier/NLL have already risen and
  ECE roughly doubles, so the **calibration-safe link budget is far tighter than
  the accuracy-safe one**.
- A differentiable **quantum Grad-CAM** gives explanations whose faithfulness vs.
  the classical head is mixed (better deletion-AUC, tied insertion-AUC).

## Repository layout
```
experiment.py        Single-run script: reproduces every table and figure
requirements.txt     Python dependencies
figures/             Figures produced by the run (in the paper)
paper/               LaTeX source (main.tex), IEEEtran.cls, compiled paper.pdf
LICENSE              MIT
```

## Setup
```bash
pip install -r requirements.txt
```

## Data
Retinal OCT-C8 (Kaggle, doi:10.34740/KAGGLE/DSV/2736749). Arrange as:
```
RetinalOCT_Dataset/
  train/<class>/*.jpeg
  test/<class>/*.jpeg
```
Only the three classes present in the training split (AMD, CNV, CSR) are used.

## Run
```bash
python experiment.py --data_dir /path/to/RetinalOCT_Dataset
```
GPU is auto-detected (uses a 1500/500 subsample); CPU uses 900/300. The script
prints Tables I–VI and the same-feature baselines, and writes all figures to
`figures/`. Random seeds are fixed for reproducibility.

## Notes
- Quantum circuits run in PennyLane (`default.qubit` for clean training,
  `default.mixed` for noisy inference). Channel noise is applied **only at
  inference**, once after encoding; the head is trained on clean data.
- The simulated channel is a simplified single-qubit abstraction, not a
  validated network protocol.

  

```
