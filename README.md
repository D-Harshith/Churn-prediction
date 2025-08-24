# Churn Analysis

## Overview
This project focuses on analyzing customer churn data from a telecommunications company. The goal is to predict whether a customer will churn (leave the service) based on various features such as demographic information, service usage, and billing details. The project includes exploratory data analysis (EDA), preprocessing, and building machine learning models to evaluate performance.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Dependencies](#dependencies)
3. [Project Structure](#project-structure)
4. [Steps to Run the Code](#steps-to-run-the-code)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Preprocessing](#preprocessing)
7. [Modeling](#modeling)
8. [Results](#results)
9. [Contributing](#contributing)

---

## Dataset
The dataset used in this project is stored in the file `WA_Fn-UseC_-Telco-Customer-Churn.csv`. It contains 7043 rows and 21 columns, including:
- **Features**: Customer demographics, services subscribed, and billing information.
- **Target Variable**: `Churn` (Yes/No) indicating whether the customer has left the service.

The dataset includes both numerical and categorical features. A detailed description of the columns can be found in the EDA section.

---

## Dependencies
To run this project, you need the following Python libraries installed:

```bash
numpy
pandas
seaborn
matplotlib
scipy
scikit-learn
```

You can install these dependencies using `pip`:

```bash
pip install numpy pandas seaborn matplotlib scipy scikit-learn
```

---

## Project Structure
The project is organized as follows:
```
.
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── churn_analysis.ipynb                   # Jupyter Notebook with the analysis
├── README.md                              # This file
└── requirements.txt                       # List of dependencies
```

---

## Steps to Run the Code
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies**:
   Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Open the Jupyter Notebook `churn_analysis.ipynb` and execute the cells sequentially:
   ```bash
   jupyter notebook churn_analysis.ipynb
   ```

4. **Explore Results**:
   After running the notebook, you will see the results of the EDA, preprocessing, and model evaluation.

---

## Exploratory Data Analysis (EDA)
The EDA phase includes:
1. **Data Overview**:
   - Basic statistics and missing value checks.
   - Identification of numerical and categorical columns.

2. **Distribution of Features**:
   - Visualizations of categorical and numerical feature distributions.
   - Relationships between features and the target variable (`Churn`).

3. **Key Observations**:
   - Categorical variables significantly influence churn.
   - Numerical features are not normally distributed.
   - The target variable (`Churn`) is imbalanced, requiring attention during modeling.

---

## Preprocessing
The preprocessing pipeline handles missing values, encodes categorical variables, and scales numerical features:
1. **Numerical Features**:
   - Missing values are imputed using the mean.
   - Features are scaled using `StandardScaler`.

2. **Categorical Features**:
   - Missing values are imputed using the most frequent value.
   - Features are encoded using `OrdinalEncoder`.

3. **Pipeline**:
   A `ColumnTransformer` is used to apply different transformations to numerical and categorical features.

---

## Modeling
Several machine learning models are trained and evaluated:
1. **Models**:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

2. **Hyperparameter Tuning**:
   - GridSearchCV is used to optimize hyperparameters for each model.
   - Evaluation metrics include Accuracy, Precision, Recall, and F1 Score.

3. **Evaluation**:
   - Models are compared based on their performance metrics.
   - Results are visualized using bar plots.

---

## Results
The best-performing model is identified based on the F1 Score. Key findings include:
- Gradient Boosting achieved the highest F1 Score of **0.8775**.
- Logistic Regression is slightly less effective compared to Random Forest.

---
