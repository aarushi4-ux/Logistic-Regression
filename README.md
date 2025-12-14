# Logistic Regression on Social Network Ads Dataset

## 📌 Project Overview
This project demonstrates the implementation of **Logistic Regression**, a supervised machine learning algorithm used for **binary classification**, on the *Social Network Ads* dataset. The goal is to predict whether a user will **purchase a product** based on demographic features such as **Gender, Age, and Estimated Salary**.

The project covers data preprocessing, model training, evaluation, and interpretation of results using standard performance metrics.

---

## 📂 Dataset Description
- **Dataset Name:** Social_Network_Ads.csv  
- **Source:** Kaggle  
- **Target Variable:** `Purchased` (0 = Not Purchased, 1 = Purchased)

### Features Used
| Feature | Description |
|--------|-------------|
| Gender | User gender (Male/Female) |
| Age | Age of the user |
| EstimatedSalary | Estimated annual salary |
| Purchased | Purchase decision (Target variable) |

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## ⚙️ Project Workflow
1. Load and explore the dataset
2. Encode categorical variables
3. Split the dataset into training and testing sets
4. Perform feature scaling
5. Train the Logistic Regression model
6. Evaluate the model using accuracy, confusion matrix, and classification report
7. Interpret results and draw conclusions

---

## 🧠 Machine Learning Model
- **Algorithm:** Logistic Regression  
- **Type:** Binary Classification  
- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 🧪 Model Results
- **Accuracy:** 88%
- The model correctly identifies most non-buyers and buyers.
- High precision for buyers indicates reliable purchase predictions.
- Slightly lower recall for buyers suggests some potential customers are missed.


