## Heart Disease Prediction & Machine Learning Assignment
End-to-end ML pipeline &amp; Streamlit app for heart disease prediction using clinical patient data. Evaluates, compares &amp; deploys 5 supervised models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest). Features comprehensive evaluation across Accuracy, ROC-AUC, Precision, Recall, F1-Score &amp; Matthews Correlation Coefficient (MCC).


### a. Problem Statement
Heart disease remains one of the critical health conditions requiring prompt and accurate diagnosis. This project evaluates five standard machine learning classification models on clinical patient parameters to predict binary heart disease status. The models are made available interactively via a Streamlit web application.

### b. Dataset Description
- **Source:** Heart Disease Dataset (`heart_disease.csv`)
- **Instances:** 1,024 samples
- **Features:** 13 predictive clinical variables:
  1. `age`: Age in years
  2. `sex`: Gender (1 = male; 0 = female)
  3. `cp`: Chest pain type (0 to 3)
  4. `trestbps`: Resting blood pressure (mm Hg)
  5. `chol`: Serum cholesterol (mg/dl)
  6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
  7. `restecg`: Resting ECG results (0, 1, 2)
  8. `thalach`: Maximum heart rate achieved
  9. `exang`: Exercise-induced angina (1 = yes; 0 = no)
  10. `oldpeak`: ST depression induced by exercise
  11. `slope`: Slope of peak exercise ST segment
  12. `ca`: Number of major vessels colored by fluoroscopy (0-3)
  13. `thal`: Thalassemia defect type
- **Target Variable:** `target_binary` (0 = No Heart Disease, 1 = Heart Disease)

### c. GitHub Repository Link
- **URL:** [https://github.com/codewithshalini/heart-disease-ml-streamlit](https://github.com/codewithshalini/heart-disease-ml-streamlit)

### d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8585 | 0.9251 | 0.8571 | 0.8298 | 0.8432 | 0.7147 |
| Decision Tree | 0.7415 | 0.7368 | 0.7356 | 0.6809 | 0.7072 | 0.4775 |
| K-Nearest Neighbor | 0.8439 | 0.9020 | 0.8690 | 0.7766 | 0.8202 | 0.6864 |
| Naive Bayes | 0.8439 | 0.9131 | 0.8298 | 0.8298 | 0.8298 | 0.6856 |
| Random Forest (Ensemble) | 0.8488 | 0.9325 | 0.8316 | 0.8404 | 0.8360 | 0.6957 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved the highest accuracy (85.85%) and MCC score (0.7147), demonstrating strong linear separability across scaled features. |
| Decision Tree | Exhibited lower performance (74.15% Accuracy) due to unconstrained split variance, leading to potential overfitting on training splits. |
| K-Nearest Neighbor | Produced high precision (86.90%) when coupled with feature standardization, effectively grouping nearest physiological profiles. |
| Naive Bayes | Delivered reliable performance (84.39% Accuracy, 0.9131 AUC) with balanced precision and recall despite feature independence assumptions. |
| Random Forest (Ensemble) | Highest overall AUC score (0.9325) and top Recall score (84.04%), showing superior sensitivity in capturing heart disease cases. |
| **Overall Winner for your dataset?** | **Logistic Regression** is the overall winner for overall overall accuracy and MCC balance, while **Random Forest** is optimal if prioritizing maximum AUC and medical recall. |
