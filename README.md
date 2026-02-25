# 🧬 HematoVision-AI

### Advanced Blood Cell Classification Using Transfer Learning

HematoVision-AI is a deep learning–powered medical imaging system that automatically classifies white blood cells from microscopic images using transfer learning. The project leverages a pretrained EfficientNet convolutional neural network to deliver accurate, scalable, and real-time blood cell classification suitable for clinical, research, and educational use.

---

## 🚀 Project Highlights

* Automated blood cell classification system
* Transfer learning with EfficientNet architecture
* High-accuracy multi-class prediction
* Real-time image inference support
* Modular and scalable design
* Industry-style training pipeline

---

## 🧠 Target Cell Classes

The model classifies the following white blood cell types:

* Eosinophil
* Lymphocyte
* Monocyte
* Neutrophil

---

## 🏗 Model Architecture

**Backbone:** EfficientNetB0 (ImageNet pretrained)
**Pipeline:**

1. Feature extraction phase
2. Fine-tuning phase

**Layers**

* EfficientNet base
* GlobalAveragePooling
* Dropout (regularization)
* Dense layer (ReLU)
* Softmax output

**Training Setup**

* Loss: Categorical Crossentropy
* Optimizer: Adam
* Technique: Transfer Learning

---

## 📂 Project Structure

```
HematoVision-AI
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── Document/
├── Project Files/
└── test.jpeg
```

---

## ⚙️ Installation

Clone repository:

```
git clone https://github.com/Phaneendra2005/HematoVision-AI.git
cd HematoVision-AI
```

Create virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Train Model

```
python train.py
```

---

## 🔍 Run Prediction

```
python predict.py --image test.jpg
```

---

## 📊 Expected Performance

| Stage            | Accuracy |
| ---------------- | -------- |
| Initial Training | ~60%     |
| Fine-Tuning      | 85–95%   |

---

## 🧪 Technologies Used

* Python
* TensorFlow / Keras
* EfficientNet
* NumPy
* OpenCV
* Deep Learning
* Transfer Learning

---

## 🎯 Real-World Applications

✔ Automated pathology diagnostics
✔ Telemedicine support systems
✔ AI medical assistants
✔ Laboratory automation tools
✔ Medical training platforms

---

## 📦 Dataset

Dataset used for training is publicly available.

Download here:
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

> Note: Dataset is not included in repo due to GitHub size limits.

---

## 🔮 Future Improvements

* Web application deployment
* Live microscope feed integration
* Mobile application interface
* Multi-class abnormal cell detection
* Model optimization for edge devices

---

## 👨‍💻 Author

**Phaneendra K**
AI Developer | Machine Learning Enthusiast

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository
🍴 Fork it
📢 Share it

---

## 📜 License

This project is open-source and available under the MIT License.
