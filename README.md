# 🧠Early Detection of Neurological Disorders

> **Team DermAlert** · PeaceOfCode Hackathon · Nirvana 2026 · Track 2

---

## 📄 Open `dermAlert.pdf` for all important information & guide

The PDF contains the **complete project documentation**, including:

- Full problem statement & task breakdown
- Model architectures and working theory
- All results, metrics, confusion matrices, and ROC curves
- Comparison across all three approaches
- Conclusions and future work

---

## 🗂️ Project Structure

```
CerebroAI/
├── dermAlert.pdf                          ← 📌 START HERE — Full guide & results
├── README.md                              ← This file
│
├── notebooks/
│   ├── Cerebroai_brain_2D_CNN.ipynb           # Approach 1: ResNet-18 + SVM
│   ├── Cerebroai_brain_3D_PCA_SVM.ipynb       # Approach 2: 3D CNN + PCA + SVM
│   ├── Cerebroai_brain_Custom_CNN_binary.ipynb    # Approach 3: Custom CNN (Binary)
│   └── Cerebroai_brain_Custom_CNN_multiclass.ipynb # Approach 3: Custom CNN (Multi)
│
└── reports/
    ├── CerebroAI_Evaluation_Report_2D_CNN.pdf
    └── CerebroAI_Evaluation_Report_3D_PCA_SVM.pdf
```

---

## 🚀 Quick Start (Google Colab)

1. Upload `MRI.zip` to your Google Drive
2. Upload `MRI_metadata.csv` when prompted
3. Open the desired notebook from the `notebooks/` folder
4. Run all cells top to bottom — GPU runtime recommended (T4)

---

## 🔬 Task Overview

| Task | Description | Threshold |
|------|-------------|-----------|
| Task 1 | MRI Preprocessing Pipeline | — |
| Task 2 | Binary Classification: CN vs AD | Bal. Accuracy > 0.91 |
| Task 3 | Multi-Class: CN vs MCI vs AD | Bal. Accuracy > 0.55 |
| Task 4 | Web-Based Application Deployment | — |

---

## 🤖 Models at a Glance

### Approach 1 — 2D CNN (ResNet-18 + SVM)
- Pretrained ResNet-18 backbone for feature extraction
- Global Average Pooling → 512-dim vector → SVM (RBF kernel)
- **Binary:** Bal. Acc 0.68 · AUC 0.76
- **Multi-class:** Bal. Acc 0.50 · AUC 0.70

### Approach 2 — 3D CNN + PCA + SVM
- Full volumetric MRI processing (all DICOM slices)
- PCA (n=50) for dimensionality reduction before SVM
- **Binary:** Bal. Acc 0.71 · AUC 0.81 ✅ Best SVM binary
- **Multi-class:** Bal. Acc 0.58 · AUC 0.78 ✅ Threshold PASSED

### Approach 3 — Custom CNN (from Scratch)
- 3× Conv2D + MaxPool blocks, Dense(128), Dropout(0.4), Softmax
- Adam optimizer · Data augmentation · EarlyStopping
- **Binary:** ~84% accuracy · Macro F1 0.81 🏆 Best binary overall
- **Multi-class:** Bal. Acc 0.44 · Macro F1 0.42

---

## 📊 Results Summary

| Approach | Task | Bal. Accuracy | AUC | Macro F1 | Pass? |
|----------|------|:---:|:---:|:---:|:---:|
| 2D CNN (ResNet-18) | Binary CN/AD | 0.675 | 0.76 | 0.657 | ✗ |
| 2D CNN (ResNet-18) | Multi CN/MCI/AD | 0.497 | 0.70 | 0.444 | ✗ |
| 3D CNN+PCA+SVM | Binary CN/AD | 0.710 | 0.81 | 0.718 | ✗ |
| **3D CNN+PCA+SVM** | **Multi CN/MCI/AD** | **0.584** | **0.78** | **0.576** | **✓** |
| **Custom CNN** | **Binary CN/AD** | **~0.80** | **—** | **0.81** | **✓** |
| Custom CNN | Multi CN/MCI/AD | 0.44 | — | 0.42 | ✗ |

---

## 🛠️ Dependencies

```bash
pip install pydicom tensorflow scikit-learn opencv-python pandas numpy matplotlib
```

> All notebooks are designed to run on **Google Colab** with GPU acceleration.

---

## 👥 Team

**Team DermAlert** — PeaceOfCode Hackathon · Nirvana 2026

---

> 📌 **For full details, evaluation reports, model architecture diagrams, and visual results — open `dermAlert.pdf`**
