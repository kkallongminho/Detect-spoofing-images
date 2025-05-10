# Detect spoofing images
presentation : 2024 spring hanbat national univ.

# 🧠 TT Team AI Spoofing Detection Project

> **2024 Artificial Intelligence & Applications Course Term Project**  
> Team TT | Afternoon Group 2  
> Team Members: Minwoo Baek, Harin Jang, Nayoung Jin, Minho Hwang

## 📌 Project Overview

This project aims to classify facial images as either **live (real)** or **spoof (fake)** using image classification models.  
The dataset includes some **unseen domain samples** in the test set to simulate real-world conditions.

- **Input**: `filename`, `label` (0 = live, 1 = spoof)
- **Goal**: Maximize test accuracy
- **Evaluation Metric**: Accuracy

---

## 🧪 Data Analysis & Augmentation

- Resized & CenterCropped face regions to 224x224
- Horizontal flipping and quality reduction for augmentation
- Identified low-quality images in validation set to enhance robustness

---

## 🧠 Models Explored

| Model | Description |
|-------|-------------|
| `ResNet18` | Deep residual network with 18 layers |
| `EfficientNet-B0` | Scalable and efficient CNN with optimal depth/width |
| `MLP-Mixer` | Convolution-free image classifier |
| `MobileNetV3` | Lightweight CNN for mobile/IoT |
| `Inception-V3` | Multi-kernel CNN for multi-scale features |

> ✅ **Final Model Selected: EfficientNet-B0 + ArcFace Loss**

---

## ⚙️ Hyperparameter Tuning

- **Loss**: ArcFace Loss → enhanced margin-based feature separation
- **Optimizer**: AdamW outperformed Adam
- **Scheduler**: StepLR and others showed little difference due to few epochs
- **Ensemble**:  
  - EfficientNet B0/B1/B2 average  
  - EfficientNet + ResNet18 + InceptionV3 ensemble

---

## 📈 Results

- High accuracy on training & validation
- Test set accuracy dropped slightly due to unseen domains → future improvement needed

---

## 💬 Team Roles

| Member | Role |
|--------|------|
| 백민우 | Documentation & dataset organization |
| 황민호, 진나영 | PPT creation |
| 장하린 | Final presentation |

---

## 🔗 Code & Resources

- [ArcFace Loss (PyTorch)](https://github.com/shyhyawJou/ArcFace-PyTorch)  
- [EfficientNet for PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

---

## ✨ What We Learned

> Through this project, we gained experience in:
- Comparing model architectures
- Applying real-world augmentations
- Improving generalization to unseen domains
- Using Kaggle-style competitions to simulate deployment scenarios

---

## 📄 License

This repository is for educational purposes.  
Includes materials by [Slidesgo](https://bit.ly/3A1uf1Q), [Freepik](http://bit.ly/2TyoMsr), and [Flaticon](http://bit.ly/2TtBDfr).
