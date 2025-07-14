# Credit_card_fraud_detection

# Credit Card Fraud Detection Using Deep Learning

This project demonstrates how to build and evaluate a deep neural network to detect credit card fraud using the highly imbalanced [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It also explores techniques like **undersampling** and **SMOTE** to handle class imbalance.

---

## 📂 Dataset

The dataset contains transactions made by European credit cardholders in September 2013. It presents **284,807 transactions**, with only **492 (0.17%) fraudulent** ones.

- Features `V1` to `V28` are PCA-transformed components.
- `Amount` and `Time` are raw.
- `Class`: 1 = Fraud, 0 = Normal.

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
    - Standardize `Amount`
    - Drop `Time` column
    - Separate features and labels

2. **Train/Test Split**
    - 70% training, 30% test split
    - Convert to NumPy arrays for Keras compatibility

3. **Model Building (Keras)**
    - Deep neural network:
      - Dense(16) → Dense(24) → Dropout(0.5) → Dense(20) → Dense(24) → Dense(1)
    - Activation: `ReLU` for hidden layers, `Sigmoid` for output
    - Loss: `Binary Crossentropy`
    - Optimizer: `Adam`

4. **Evaluation (on imbalanced data)**
    - High accuracy, but poor fraud recall due to imbalance

5. **Confusion Matrix Visualization**
    - With `sklearn.metrics.confusion_matrix` and `matplotlib`

6. **Undersampling**
    - Equal sample of fraud and non-fraud
    - Retrain model and compare results

7. **SMOTE (Synthetic Minority Oversampling Technique)**
    - Use `imbalanced-learn` to oversample minority class
    - Retrain and evaluate again

---

## 📈 Results

| Method         | Accuracy | True Positives | False Negatives | Comments                         |
|----------------|----------|----------------|------------------|----------------------------------|
| Imbalanced     | ~99.94%  | 111            | 36               | Poor recall due to imbalance     |
| Undersampling  | ~93.4%   | 131            | 14               | Better recall, lower accuracy    |
| SMOTE          | ~99.8%   | 84,905         | 6                | Best balance of recall and acc.  |

---

## 📊 Confusion Matrix Samples

**Imbalanced Dataset**
[[85281 15]
[ 36 111]]

**Undersampled Dataset**
[[151 0]
[ 14 131]]

**SMOTE Oversampling**
[[85343 335]
[ 6 84905]]


---

## 🛠️ Setup Instructions

### Environment

- Python 3.6+
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy, Matplotlib
- imbalanced-learn

### Installation

```bash
pip install numpy pandas matplotlib scikit-learn keras imbalanced-learn
