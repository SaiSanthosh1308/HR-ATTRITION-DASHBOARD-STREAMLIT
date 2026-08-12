# 👥 HR Attrition Dashboard (Streamlit)

An interactive dashboard for exploring employee attrition patterns and predicting flight risk, built with **Python + Streamlit** on the IBM HR Analytics dataset.

🔗 **Live demo:** [https://hr-attrition-dashboard-app-dkexz9bnk5avbasssgpfnd.streamlit.app/]

## Features

**Explore tab**
- Interactive filters — department, gender, job role, age range, attrition status
- KPI cards — headcount, attrition rate, average monthly income, average tenure
- Attrition by department and job role
- Monthly income vs attrition, age distribution by attrition
- Job satisfaction vs work-life balance bubble chart
- Correlation heatmap across numeric features

**Predict tab**
- Choose between Logistic Regression or Random Forest
- Train the model live and view accuracy, precision, recall, F1 score
- Confusion matrix and ROC curve (AUC)
- Top feature importances driving attrition (Random Forest)

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Plotly
- Scikit-learn

## Run it locally

```bash
git clone https://github.com/SaiSanthosh1308/HR-ATTRITION-DASHBOARD-STREAMLIT.git
cd HR-ATTRITION-DASHBOARD-STREAMLIT
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

1,470 employee records with 35 features covering demographics, compensation, job role, satisfaction scores, work-life balance, and tenure, labeled with attrition (Yes/No). Source: IBM HR Analytics Employee Attrition dataset.

## About This Project

Part of a growing portfolio of interactive analytics dashboards built as I work toward a data analyst role — combining exploratory analysis with a live, trainable prediction model for a classic HR business problem.

## 📸 Dashboard Preview

| **Explore Tab** |
| :---: |
| Explore Tab Preview <img width="1400" height="1200" alt="explore" src="https://github.com/user-attachments/assets/e2786f17-0428-4058-bc7d-ef2a27b10eff" /> |
| **Predict Tab** |
| Predict Tab Preview <img width="1400" height="1200" alt="predict" src="https://github.com/user-attachments/assets/026b29ac-7d10-4a2d-9114-9cd1b15a3c9f" /> |

