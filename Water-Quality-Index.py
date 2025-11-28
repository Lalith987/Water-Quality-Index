import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import warnings
from pandas.errors import SettingWithCopyWarning

# Suppress SettingWithCopyWarning for cleaner output
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


# Load and clean the dataset
# Make sure your corrected waterQuality.csv file is in the same folder
data = pd.read_csv('waterQuality.csv')

# Remove rows where the target ('Potability') is NaN
data.dropna(subset=['Potability'], inplace=True)

print("Dataset loaded and cleaned successfully!\n - water_quality_project.py:28")


# Basic EDA
print("Shape: - water_quality_project.py:32", data.shape)
print("\nData Info: - water_quality_project.py:33")
data.info()
print("\nMissing values:\n - water_quality_project.py:35", data.isnull().sum())
print("\nFirst 5 rows:\n - water_quality_project.py:36", data.head())


# Feature selection and preprocessing
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
            'Organic_carbon', 'Trihalomethanes', 'Turbidity']
target = 'Potability'

X = data[features]
y = data[target]

# Handle missing values in features (fill with mean)
X.fillna(X.mean(), inplace=True)

# Calculate the scale_pos_weight for XGBoost to handle class imbalance
# This is the ratio of negative class to positive class
scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]


# Train-test split (with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Define models & hyperparameters with imbalance handling
models = {
    "Logistic Regression": {
        "model": LogisticRegression(class_weight='balanced'),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["liblinear"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(class_weight='balanced'),
        "params": {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(class_weight='balanced'),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ["uniform", "distance"]
        }
    },
    "SVM": {
        "model": SVC(class_weight='balanced'),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        }
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss'),
        "params": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        }
    }
}


# Train, tune, and evaluate models
results = []
for name, m in models.items():
    print(f"\nTraining & Tuning: {name} - water_quality_project.py:124")
    # FIX: Changed scoring to f1_weighted for better evaluation on imbalanced data
    clf = GridSearchCV(m['model'], m['params'], cv=5, n_jobs=-1, scoring='f1_weighted')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Best Params: {clf.best_params_} - water_quality_project.py:132")
    print(f"Accuracy: {acc:.4f} - water_quality_project.py:133")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    results.append({
        "Model": name,
        "Best_Params": clf.best_params_,
        "Accuracy": acc
    })


# Compare models
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
print("\n Model Comparison \n - water_quality_project.py:155")
print(results_df)


# Feature Importance (from the best model)
best_model_name = results_df.iloc[0]['Model']
print(f"\n Feature Importance from {best_model_name} - water_quality_project.py:161")

if best_model_name in models:
    best_model_info = models[best_model_name]
    final_model = best_model_info['model'].set_params(**results_df.iloc[0]['Best_Params'])
    final_model.fit(X_train, y_train)

    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        plt.figure(figsize=(10, 6))
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f"Feature Importance ({best_model_name})")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
    elif hasattr(final_model, 'coef_'): # For Logistic Regression
        importances = final_model.coef_[0]
        plt.figure(figsize=(10, 6))
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f"Feature Importance ({best_model_name})")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()
    else:
        print(f"{best_model_name} does not directly support feature importance. - water_quality_project.py:191")
else:
    print("Could not retrieve the best model for feature importance. - water_quality_project.py:193")


#  Report Section
print("\n\n Water Quality Prediction Report \n - water_quality_project.py:197")
print("Dataset: Water Quality Dataset - water_quality_project.py:198")
print("Objective: Predict water potability (safe to drink) using water quality parameters.\n - water_quality_project.py:199")
print("Steps Taken: - water_quality_project.py:200")
print("1. Data loading and cleaning (removed rows with missing target values). - water_quality_project.py:201")
print("2. Exploratory Data Analysis (EDA) to understand data shape and missing values. - water_quality_project.py:202")
print("3. Feature selection and preprocessing (imputed missing feature values with mean). - water_quality_project.py:203")
print("4. Performed a stratified split into training/testing sets and applied feature scaling. - water_quality_project.py:204")
print("5. Trained and evaluated 7 different classification models, using class weights to handle imbalance. - water_quality_project.py:205")
print("6. Performed hyperparameter tuning for each model using GridSearchCV, optimizing for 'f1_weighted' score. - water_quality_project.py:206")
print("7. Evaluated models based on Accuracy, Precision, Recall, and F1score. - water_quality_project.py:207")
print("8. Visualized performance with Confusion Matrices. - water_quality_project.py:208")
print("9. Analyzed feature importance from the bestperforming model.\n - water_quality_project.py:209")

print("Conclusion: - water_quality_project.py:211")
print(f"The best performing model was '{results_df.iloc[0]['Model']}' with an accuracy of {results_df.iloc[0]['Accuracy']:.4f}. - water_quality_project.py:212")
print("By using class weights and optimizing for F1score, the models are now better at identifying the rare 'potable' class. - water_quality_project.py:213")
print("Feature importance analysis helps identify the most impactful parameters on water quality. - water_quality_project.py:214")