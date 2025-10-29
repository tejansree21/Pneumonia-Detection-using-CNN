# 🫁 Pneumonia Detection using CNN

## 📘 Overview
This project uses **Deep Learning** to detect **Pneumonia** from **Chest X-ray images**.  
The model is trained on the popular **Chest X-Ray (Pneumonia) dataset** and can distinguish between **Normal** and **Pneumonia-affected lungs** with high accuracy.

---

## 🧩 Dataset
**Dataset Source:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Dataset Structure:**

**Classes:**
- `NORMAL` → Healthy lungs  
- `PNEUMONIA` → Infected lungs  

---

## 🧠 Model Architecture
| Layer Type | Parameters |
|-------------|-------------|
| Conv2D (32 filters) | ReLU + MaxPooling |
| Conv2D (64 filters) | ReLU + MaxPooling |
| Conv2D (128 filters) | ReLU + MaxPooling |
| Dense (128 neurons) | ReLU + Dropout (0.5) |
| Output | Sigmoid Activation (Binary) |

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy  

---

## 📈 Results
| Metric | Value |
|:--------|:------:|
| Training Accuracy | ~96% |
| Validation Accuracy | ~93% |
| Test Accuracy | ~92% |

*(Performance may vary depending on training epochs and hardware)*

---

## 🧪 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/pneumonia-detection.git
   cd pneumonia-detection
