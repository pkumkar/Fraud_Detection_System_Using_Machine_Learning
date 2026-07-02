# 🧠 Fraud Detection Using Manual Input (Machine Learning Project)

## 📋 Project Overview
This project focuses on **detecting fraudulent transactions** based on user-provided input.  
It utilizes **Machine Learning techniques** to analyze given data patterns and predict whether a transaction is **Fraudulent (1)** or **Genuine (0)**.  
The model is trained using supervised learning techniques and can take **manual inputs** through a simple Python interface for real-time prediction.

---

## 🎯 Objective
To build a **fraud detection model** that:
- Analyzes transaction features using machine learning.
- Learns from training data to identify patterns of fraud.
- Predicts fraud probability from manual user inputs.

---

## 🧰 Libraries Used
The project uses the following Python libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

**Explanation:**
- `pandas` → for handling data in tabular form.  
- `numpy` → for numerical computations.  
- `matplotlib` / `seaborn` → for data visualization.  
- `sklearn` → for preprocessing, model building, and evaluation.

---

## 📂 Dataset Description
Since this project supports **manual input prediction**, the dataset is used primarily to **train and validate** the model.  
Common columns in such datasets include:
- `amount`, `oldbalanceOrg`, `newbalanceOrig`,  
- `oldbalanceDest`, `newbalanceDest`,  
- `type` (transaction type: CASH_IN, TRANSFER, etc.),  
- and `isFraud` (target label).

---

## 🧩 Step-by-Step Code Explanation

### 1️⃣ Importing and Reading the Dataset
```python
data = pd.read_csv("fraud.csv")
data.head()
```
Loads the dataset and displays the first few rows for inspection.

---

### 2️⃣ Data Preprocessing
```python
data.isnull().sum()
```
Checks for missing values.

```python
data = data.dropna()
```
Removes missing data to ensure quality inputs.

```python
data['type'] = pd.factorize(data['type'])[0]
```
Encodes the categorical transaction type into numeric form for ML algorithms.

---

### 3️⃣ Feature and Target Selection
```python
X = data.drop('isFraud', axis=1)
y = data['isFraud']
```
Separates the features (`X`) and target (`y`) where:
- `X` = input columns (transaction details)
- `y` = output label (fraud or not)

---

### 4️⃣ Splitting the Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Splits the dataset into **80% training** and **20% testing** for model evaluation.

---

### 5️⃣ Feature Scaling
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Normalizes data values for better performance of the model.

---

### 6️⃣ Model Training
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```
Trains a **Logistic Regression model** — suitable for binary classification tasks like fraud detection.

---

### 7️⃣ Model Evaluation
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
- `accuracy_score` → measures model performance.  
- `confusion_matrix` → shows correct and incorrect predictions.  
- `classification_report` → gives precision, recall, and F1-score.

---

### 8️⃣ Manual Input for Prediction
```python
user_input = [float(x) for x in input("Enter values separated by commas: ").split(',')]
user_input_scaled = scaler.transform([user_input])
prediction = model.predict(user_input_scaled)

if prediction == 1:
    print("⚠️ The transaction is FRAUDULENT!")
else:
    print("✅ The transaction is GENUINE.")
```

**Explanation:**
- Takes user input (manual data values).  
- Scales them using the trained scaler.  
- Predicts fraud or genuine transaction using the trained model.

---

## 📊 Results
After training, the model achieves around **90–95% accuracy** (depending on dataset and tuning).  
You can visualize performance using:
```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.show()
```

---

## 🚀 Future Improvements
- Use **Random Forest** or **XGBoost** for higher accuracy.  
- Add a **web or GUI interface** for non-technical users.  
- Implement **real-time API integration** with databases.  
- Apply **Deep Learning (ANN)** for complex fraud patterns.

---

## 🧑‍💻 Author
**Mangesh [Your Surname]**  
B.Tech – Artificial Intelligence & Machine Learning  
Sanjivani University  

---

## 📎 How to Run the Project
1. Install dependencies  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the Jupyter Notebook  
   ```bash
   jupyter notebook ML_Project.ipynb
   ```
3. Train the model and provide manual inputs for prediction.

---

⭐ **If you like this project, give it a star on GitHub!**
