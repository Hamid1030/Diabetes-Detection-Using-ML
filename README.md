# Diabetes Prediction using Support Vector Machine (SVM)

A machine learning project that predicts whether a person is diabetic based on diagnostic measurements from the **Pima Diabetes Dataset**.

The project uses a **Support Vector Machine (SVM)** classifier with a linear kernel and standardizes the input features before training and prediction.

## 📌 Project Overview

Diabetes is a chronic disease that affects how the body regulates blood glucose. Machine learning can be used to analyze medical measurements and estimate whether a person is likely to have diabetes.

In this project:

* The dataset is loaded using Pandas.
* The data is explored and analyzed.
* Features and labels are separated.
* Features are standardized using `StandardScaler`.
* The dataset is divided into training and testing sets.
* A linear **Support Vector Machine (SVM)** classifier is trained.
* The model is evaluated using accuracy.
* A predictive system is created to classify new patient data.

---

## 📊 Dataset

The project uses the **Pima Diabetes Dataset**, which contains **768 records** and **9 columns**.

The dataset contains 8 input features and 1 target variable:

| Feature                    | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `Pregnancies`              | Number of pregnancies                           |
| `Glucose`                  | Plasma glucose concentration                    |
| `BloodPressure`            | Diastolic blood pressure                        |
| `SkinThickness`            | Triceps skin fold thickness                     |
| `Insulin`                  | 2-Hour serum insulin                            |
| `BMI`                      | Body Mass Index                                 |
| `DiabetesPedigreeFunction` | Diabetes pedigree function                      |
| `Age`                      | Age of the patient                              |
| `Outcome`                  | Target variable: 0 = Non-diabetic, 1 = Diabetic |

### Dataset Distribution

The target variable contains:

* **500** non-diabetic cases (`Outcome = 0`)
* **268** diabetic cases (`Outcome = 1`)

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Jupyter Notebook

### Python Libraries

```python
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

---

## 🔄 Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   ↓
Data Collection
   ↓
Data Analysis
   ↓
Feature / Label Separation
   ↓
Feature Standardization
   ↓
Train-Test Split
   ↓
SVM Model Training
   ↓
Model Evaluation
   ↓
Prediction on New Data
```

---

## 🔍 Data Analysis

The dataset is first loaded using Pandas:

```python
diabetes_dataset = pd.read_csv('diabetes.csv')
```

The first five rows can be viewed using:

```python
diabetes_dataset.head()
```

The dataset dimensions are:

```text
(768, 9)
```

Statistical information is obtained using:

```python
diabetes_dataset.describe()
```

The distribution of the target variable can be checked with:

```python
diabetes_dataset['Outcome'].value_counts()
```

---

## 🎯 Separating Features and Labels

The `Outcome` column is used as the target variable.

```python
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
```

Where:

* `X` → Input features
* `Y` → Target/output labels

There are **8 input features** and **1 target variable**.

---

## ⚖️ Data Standardization

Since the features have different scales, `StandardScaler` is used to standardize them.

```python
scaler = StandardScaler()

scaler.fit(X)
standard_data = scaler.transform(X)

X = standard_data
Y = diabetes_dataset['Outcome']
```

Standardization transforms the features so that they have approximately:

* Mean = 0
* Standard deviation = 1

This is particularly useful for SVM models because SVMs can be sensitive to differences in feature scales.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)
```

The resulting datasets are:

| Dataset       | Samples | Features |
| ------------- | ------: | -------: |
| Full dataset  |     768 |        8 |
| Training data |     614 |        8 |
| Testing data  |     154 |        8 |

`stratify=Y` is used to maintain a similar proportion of diabetic and non-diabetic cases in both training and testing sets.

---

## 🤖 Model Training

A Support Vector Machine classifier with a linear kernel is used:

```python
classifier = svm.SVC(kernel='linear')
```

The model is trained using:

```python
classifier.fit(X_train, Y_train)
```

---

## 📈 Model Evaluation

The model is evaluated on both training and testing data.

### Training Accuracy

```text
0.7866449511400652
```

Approximately:

**78.66%**

### Testing Accuracy

```text
0.7727272727272727
```

Approximately:

**77.27%**

| Evaluation |   Accuracy |
| ---------- | ---------: |
| Training   | **78.66%** |
| Testing    | **77.27%** |

The relatively small difference between training and testing accuracy suggests that the model does not show a large amount of overfitting based on accuracy alone.

> Accuracy is only one evaluation metric. For a medical prediction problem, metrics such as precision, recall, F1-score, ROC-AUC, and confusion matrix should also be considered.

---

## 🔮 Predictive System

The trained model can be used to make predictions for new patient data.

For example:

```python
input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
```

The input is converted into a NumPy array:

```python
input1 = np.asarray(input_data)
```

The array is reshaped:

```python
input2 = input1.reshape(1, -1)
```

The input is then standardized using the same scaler used during training:

```python
std_data = scaler.transform(input2)
```

Finally, the trained SVM model makes the prediction:

```python
prediction = classifier.predict(std_data)
```

### Example 1

Input:

```text
(4, 110, 92, 0, 0, 37.6, 0.191, 30)
```

Prediction:

```text
[0]
```

Result:

```text
The person is not diabetic
```

### Example 2

Input:

```text
(5, 166, 72, 19, 175, 25.8, 0.587, 51)
```

Prediction:

```text
[1]
```

Result:

```text
The person is diabetic
```

---

## ⚠️ StandardScaler Feature-Name Warning

When predicting new data, the following warning may appear:

```text
UserWarning: X does not have valid feature names,
but StandardScaler was fitted with feature names
```

This happens because the scaler was fitted using a Pandas DataFrame containing column names:

```python
scaler.fit(X)
```

but the prediction input is provided as a NumPy array.

The model still produces a prediction, but the warning can be avoided by providing the input as a DataFrame with the same feature names.

For example:

```python
input_data = pd.DataFrame(
    [[4, 110, 92, 0, 0, 37.6, 0.191, 30]],
    columns=[
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]
)

std_data = scaler.transform(input_data)
prediction = classifier.predict(std_data)
```

This also makes the prediction pipeline clearer and less prone to accidentally using the features in the wrong order.

---

## 📁 Suggested Project Structure

```text
diabetes-prediction/
│
├── diabetes.csv
├── diabetes_prediction.ipynb
├── README.md
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetes-prediction.git
```

### 2. Navigate to the project directory

```bash
cd diabetes-prediction
```

### 3. Install the required libraries

```bash
pip install numpy pandas scikit-learn jupyter
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
diabetes_prediction.ipynb
```

and run the cells sequentially.

---

## 📦 Requirements

A `requirements.txt` file can contain:

```text
numpy
pandas
scikit-learn
jupyter
```

---

## 📌 Results

The linear SVM achieved the following results:

```text
Training Accuracy : 78.66%
Testing Accuracy  : 77.27%
```

The model can successfully classify example inputs into two categories:

```text
0 → Non-diabetic
1 → Diabetic
```
---

## 🧠 Key Concepts Demonstrated

This project demonstrates practical applications of:

* Supervised machine learning
* Binary classification
* Support Vector Machines
* Feature standardization
* Train-test splitting
* Model evaluation
* Predictive systems
* NumPy and Pandas data processing
* Scikit-learn

---

