Heart Disease Prediction  
Developed By: Zoya Zahid

This project builds a logistic regression model to predict whether a patient is at risk of heart disease, using real-world medical data and standard evaluation metrics.
---
 🎯 Task Objective
To develop a binary classification model using logistic regression that predicts heart disease based on patient health attributes.

---
Dataset Used
- Name: Heart Disease UCI Dataset  
- Source: [Kaggle – Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)  
- File Used:`heart.csv`  
- Shape: ~303 rows × 14 columns

---

Model Applied
-Logistic Regression (via `sklearn.linear_model`)
- Train/Test Split: 80/20 ratio
- Evaluation Metrics:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score and Curve

---

Key Results and Findings
- Model achieved ~85% accuracy (depending on random state)
- **Top predictive features:`cp`, `thal`, `ca`, `exang`
- No missing data found in the dataset
- ROC curve showed a strong area under the curve (AUC ≈ 0.88), indicating reliable performance

---
Tools and Libraries Used
- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib

---
Report generated: June 2025
