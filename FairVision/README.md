# 🧠 FairVision: Detecting and Mitigating Bias in CNN-Based Age Group Classification

## 📌 Project Overview

FairVision is a deep learning project focused on building a **Convolutional Neural Network (CNN)** to classify **age groups from facial images** using the FairFace dataset.

This project goes beyond standard classification by incorporating **Responsible AI principles**, including:

- Bias detection across demographic groups  
- Fairness evaluation using race and gender  
- Implementation of bias mitigation strategies  
- Analysis of performance vs fairness trade-offs  

---

## 🎯 Objective

The goal of this project is to:

- Build a CNN model to predict **age group from face images**
- Evaluate model performance across **race and gender**
- Detect potential **bias in predictions**
- Apply **mitigation strategies** to improve fairness

---

## 📂 Dataset

**Dataset Used:** FairFace (Hugging Face)

🔗 https://huggingface.co/datasets/HuggingFaceM4/FairFace

### Dataset Details

- Total Samples: ~97,698  
- Train: 86,744  
- Validation: 10,954  
- Image Size: 224 × 224  
- Configuration Used: 0.25  

### Labels

#### Age Groups (Target Variable)

- 0–2  
- 3–9  
- 10–19  
- 20–29  
- 30–39  
- 40–49  
- 50–59  
- 60–69  
- 70+  

#### Demographic Attributes (Fairness Analysis)

- Race (7 classes)
- Gender (Male, Female)

---

## ⚙️ Model Architecture

A **custom CNN is built from scratch using PyTorch** (no transfer learning).

### Architecture Overview
<center>
Input Image (224x224)   <br>
↓                       <br>
Conv Layer + ReLU       <br>
↓                       <br>
Max Pooling             <br>
↓                       <br>
Conv Layer + ReLU       <br>
↓                       <br>
Max Pooling             <br>
↓                       <br>
Fully Connected Layers  <br>
↓                       <br>
Softmax Output (9 classes)
</center>

---

## 📥 Input / Output

### Input
- Face image (JPG / PNG)
- Size: 224 × 224

### Output
- Predicted age group
- Confidence probability

### Example Output

> Prediction: 20–29 \
Confidence: 0.91


---

## 🧪 Methodology

### 1️⃣ Data Preparation

- Image resizing and normalization  
- Data augmentation (flip, rotation, brightness)  
- Train-validation split  

---

### 2️⃣ Baseline Model

- Custom CNN trained using PyTorch  
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  

---

### 3️⃣ Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- Class-wise performance  

---

### 4️⃣ Fairness Audit

Model performance evaluated across:

- Race groups  
- Gender groups  

Metrics used:

- Group-wise accuracy  
- Performance gap  
- Worst-group accuracy  

---

### 5️⃣ Bias Detection

Analysis includes:

- Identification of underperforming groups  
- Accuracy disparities across demographics  
- Interpretation of fairness gaps  

---

### 6️⃣ Bias Mitigation Techniques

Implemented strategies:

- Oversampling underrepresented groups  
- Class-weighted loss  
- Balanced mini-batch training  

---

### 7️⃣ Comparative Analysis

| Model        | Accuracy | Fairness |
|-------------|--------|---------|
| Baseline     | High   | Biased  |
| Mitigated    | Slightly Lower | Improved |

---

## 📊 Results (Example)

- Baseline Accuracy: 92%  
- Worst-group Accuracy: 75%  

After Mitigation:

- Overall Accuracy: 89%  
- Worst-group Accuracy: 85%  

✅ Improved fairness with minimal performance loss  

---

## 🚀 Demo Application

A simple **Streamlit web application** is included.

### Features:

- Upload face image  
- Predict top 3 age groups  
- Display confidence scores  
- Show system limitations  

---

## ▶️ How to Run

Install dependencies

`pip install -r requirements.txt`

Run jupyter Notebook with correct kernel


---

## 📌 Key Learnings

- CNN models can achieve high accuracy but may introduce bias  
- Fairness evaluation is essential in AI systems  
- Bias mitigation improves ethical reliability  
- There is a trade-off between accuracy and fairness  

---

## ⚠️ Limitations

- Dataset may contain inherent bias  
- Model not suitable for real-world deployment  
- Limited generalization beyond dataset  

---

## 🧭 Ethical Considerations

- Avoid misuse in surveillance systems  
- Ensure transparency in AI predictions  
- Consider societal impact of biased models  

---

## 📚 References

- FairFace Dataset (Hugging Face)  
- CNN and Deep Learning literature  
- Responsible AI research papers  

---

## 👨‍💻 Author

**Nuwanaka WAS**  \
University of Moratuwa  \
Biomedical / Electronic Engineering \
IJSE CAME 

---

## ⭐ Final Note

This project demonstrates that AI systems must be evaluated not only for accuracy but also for fairness, transparency, and ethical impact.