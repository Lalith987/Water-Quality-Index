# Water Quality Prediction using Machine Learning

## 📌 Project Overview
This project applies Machine Learning techniques to predict whether water is safe to drink (potable) based on various water quality metrics. The dataset includes parameters such as pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, and Turbidity.

The project handles class imbalance, performs extensive Exploratory Data Analysis (EDA), and compares the performance of 7 different classification algorithms.

## 📂 Dataset
The dataset used is `waterQuality.csv`. It contains water quality metrics for 3276 samples.
- **Target Variable:** `Potability` (1 = Safe, 0 = Not Safe)
- **Features:** 9 distinct water quality parameters.

## 🛠 Technologies Used
- **Python**
- **Pandas & NumPy** (Data Manipulation)
- **Matplotlib & Seaborn** (Visualization)
- **Scikit-Learn** (Preprocessing & Models)
- **XGBoost** (Advanced Gradient Boosting)

## ⚙️ Methodology
1. **Data Cleaning:** Removed null values in the target variable and imputed missing feature values with the mean.
2. **EDA:** Analyzed distributions and correlations.
3. **Preprocessing:** Applied `StandardScaler` to normalize features.
4. **Handling Imbalance:** Used `class_weight='balanced'` and `scale_pos_weight` to improve detection of the minority class.
5. **Model Training:** Trained the following models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Gradient Boosting
   - XGBoost
6. **Evaluation:** Optimized hyperparameters using `GridSearchCV` (scoring on F1-Weighted) and evaluated using Confusion Matrices and Accuracy scores.

## 📊 Results
The models were compared based on accuracy and F1 scores. The Feature Importance analysis highlights **pH**, **Sulfate**, and **Hardness** as critical factors in determining water safety.

*(You can add a screenshot of your confusion matrix here)*

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/water-quality-prediction.git](https://github.com/YOUR_USERNAME/water-quality-prediction.git)
