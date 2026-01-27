Establishing your project’s narrative and technical flow is essential for effectively communicating its value to others, whether they are colleagues or interviewers.

Below is a detailed end-to-end breakdown of your project, **BankGuard: Churn Intelligence**, including a comprehensive description and a technical pipeline flow based on your implementation.

---

### **1. Project Overview & Business Value**

**The Problem:** Customer attrition, or "churn," is a critical challenge for banking institutions. It is significantly more expensive to acquire a new customer than to retain an existing one.

**The Solution:** This project utilizes a dataset of 10,000 customers across France, Germany, and Spain to build a predictive model that identifies high-risk customers likely to close their accounts. By identifying these individuals early, the bank can apply targeted retention strategies to preserve revenue.

---

### **2. Technical Pipeline Flow**

Your project follows a rigorous, industry-standard machine learning lifecycle.

#### **A. Data Audit & Initial Cleaning**

* **Dataset Dimensions:** Started with 10,000 records and 14 features.
* **Noise Reduction:** Dropped non-predictive features like `RowNumber`, `CustomerId`, and `Surname` to prevent "data leakage" and focus solely on behavioral/demographic indicators.
* **Quality Check:** Verified there were no duplicate entries or missing values that could skew results.

#### **B. Exploratory Data Analysis (EDA)**

* **Visualizing Distributions:** Used histogram grids to understand the distribution of numeric features like Credit Score, Age, and Balance.
* **Insight Discovery:** Analyzed how specific demographics correlated with the target variable, `Exited`.

#### **C. Preprocessing & Feature Engineering**

* **Categorical Encoding:** Used `OneHotEncoder` to transform text-based features like `Geography` and `Gender` into numerical formats that models can process.
* **Numerical Scaling:** Applied `MinMaxScaler` to normalize features like `EstimatedSalary` and `Balance`, ensuring that large numbers don't disproportionately influence the model's weightings.
* **Imbalance Management (The "Pro" Step):** Because only ~20% of customers churned, the dataset was highly imbalanced. You implemented **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic data for the minority class, ensuring the model is trained to recognize churners accurately rather than just guessing "no churn" for everyone.

#### **D. Modeling & Benchmarking**

You experimented with multiple algorithms to find the most robust solution:

* **Traditional ML:** Evaluated Logistic Regression, Decision Trees, and Random Forest.
* **Advanced ML:** Tested **XGBoost** and **Support Vector Machines (SVM)**.
* **Deep Learning:** Developed a Neural Network using **TensorFlow/Keras** to find complex, non-linear patterns in the data.

#### **E. Evaluation & Deployment**

* **Metrics:** Used Confusion Matrices and Classification Reports, focusing on **Recall** to ensure at-risk customers are not missed.
* **Selection:** Determined that **Deep Learning** and **SVM** provided the most stable and balanced results across both "Stays" and "Exits".
* **Persistence:** Utilized `joblib` for model export, preparing the system for real-world deployment.

---

### **3. Long-Form Project Description for Documentation**

> "The **BankGuard: Churn Intelligence** project is a comprehensive data science study into customer retention. The core of the project involves an end-to-end pipeline that transforms raw banking data into predictive insights. A standout feature is the application of **SMOTE** to handle class imbalance, which significantly improved the model's ability to identify actual churn cases. By benchmarking multiple architectures—from **XGBoost** to **Neural Networks**—this project demonstrates a high-precision approach to solving real-world binary classification problems in the fintech space."

### **4. Summary Diagram of the Workflow**

1. **Data Intake:** Load 10k records from CSV.
2. **Cleaning:** Drop identifiers; check for duplicates.
3. **Preprocessing:** Encode categorical data; scale numerical ranges.
4. **Balancing:** Apply SMOTE to the training set.
5. **Training:** Fit SVM, Random Forest, and Deep Learning models.
6. **Evaluation:** Compare metrics via Heatmaps and Confusion Matrices.
7. **Finalize:** Export the best-performing model for production.

### **A. Logistic Regression (The Baseline)**
Logistic Regression serves as the fundamental baseline for binary classification. It is used to understand the linear relationship between customer features (like credit score) and the probability of exit.
Implementation: Built using an imbalanced-learn pipeline that combines MinMaxScaler for numerical normalization and OneHotEncoder for categorical data.
SMOTE Integration: Oversampling was applied within the pipeline to ensure the model learned from a balanced representation of both "Stay" and "Exit" classes.

![Logistic Regression Pipline](https://github.com/Ranjit933/End-to-End-Bank-Customer-Churn-Analysis/blob/main/Pipline/Logistic.png)

### **B. Decision Tree (The Intuitive Model)**
Decision Trees offer a high degree of interpretability by splitting the data into a flowchart-like structure based on feature importance.
Technique: Utilized DecisionTreeClassifier to create a non-linear boundary for customer segments.
Feature Splits: This model helps identify specific "cutoff points" (e.g., Age > 42) that significantly increase churn risk.

![Decision Tree Pipline](https://github.com/Ranjit933/End-to-End-Bank-Customer-Churn-Analysis/blob/main/Pipline/Decision%20tree.png)

### **C. Random Forest (The Ensemble Powerhouse)**
To improve upon the single Decision Tree, Random Forest uses an ensemble of multiple trees to reduce overfitting and increase prediction stability.
Strategy: Implemented RandomForestClassifier, which averages the results of many individual trees to produce a more robust final prediction.
Advantage: Highly effective at handling complex datasets with mixed feature types without requiring extensive manual tuning.

![Randome Forest Pipline](https://github.com/Ranjit933/End-to-End-Bank-Customer-Churn-Analysis/blob/main/Pipline/RandomForest.png)

4. Support Vector Machine - SVM (The Geometric Optimizer)
SVM is one of your top-performing models in this project. It works by finding the "Hyperplane" that best separates the two classes in a high-dimensional space.
Top Performance: Along with Deep Learning, your analysis showed that SVM provided the most leveled predictions for both "Stays" and "Exits".
Precision/Recall Balance: The SVM model was saved as best_svm_model.pkl for deployment because of its superior ability to generalize across new customer data.

![SVM Pipline](https://github.com/Ranjit933/End-to-End-Bank-Customer-Churn-Analysis/blob/main/Pipline/SVM.png)

5. XGBoost (The Gradient Booster)
XGBoost (Extreme Gradient Boosting) is a state-of-the-art algorithm frequently used in industry competitions for its speed and performance.
Mechanism: It builds trees sequentially, where each new tree corrects the errors made by the previous ones.
Efficiency: Integrated into the pipeline to compare how modern boosting techniques perform against traditional linear and ensemble models.

![XGBOOST Pipline](https://github.com/Ranjit933/End-to-End-Bank-Customer-Churn-Analysis/blob/main/Pipline/XGBOOST.png)


# BankGuard: Churn Intelligence 🏦

An end-to-end Machine Learning pipeline designed to identify high-risk bank customers. This project utilizes **Deep Learning** and **SVM**, optimized with **SMOTE** to handle class imbalance across 10,000 customer records.

![WorkFlow](https://github.com/Ranjit933/End-to-End-Bank-Customer-Churn-Analysis/blob/main/Pipline/Gemini_Generated_Image_n3xbagn3xbagn3xb.png)

---

## 📂 Repository Structure
```text
BankGuard-Churn-Intelligence/
├── data/                       # Raw Dataset (CSV)
├── models/                     # Saved Pickle/H5 files & Scalers
├── notebooks/                  # EDA & Model Development Notebooks
├── src/                        # Modular Python Scripts (Preprocessing/Training)
├── requirements.txt            # Environment Dependencies
└── README.md                   # Documentation


graph TD
    A[Raw Data: Churn_Modelling.csv] --> B[Data Cleaning & Feature Selection]
    B --> C{Preprocessing}
    C --> D[One-Hot Encoding: Geography/Gender]
    C --> E[Min-Max Scaling: Balance/Salary]
    D & E --> F[Dataset Split: Train/Test]
    F --> G[Class Imbalance Management: SMOTE]
    G --> H{Model Training & Benchmarking}
    H --> I[Deep Learning: Keras/TensorFlow]
    H --> J[Support Vector Machines: SVM]
    H --> K[XGBoost & Random Forest]
    I & J & K --> L[Evaluation: Recall & Confusion Matrix]
    L --> M[Deployment: Joblib Model Export]



