# Student Performance Predictor (Beginner ML Project)

A small end-to-end machine-learning demo that estimates a sample student score from study hours, attendance, previous score, assignment completion, and sleep hours.

> **Important:** `data/student_performance.csv` is synthetic practice data, not real student records. This project demonstrates an ML workflow and is not validated for real educational decisions.

## Features
- Load tabular data with Pandas
- Train/test split
- Scikit-learn pipeline with median imputation, scaling, and Linear Regression
- Evaluate regression using MAE, RMSE, and R²
- Interactive Streamlit interface
- Save the trained model and evaluation metrics locally

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies and train the model:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit prints a local URL to open in your browser.

## Project structure

```text
student-performance-ml/
├── app.py
├── train_model.py
├── data/
│   └── student_performance.csv
├── models/
│   └── .gitkeep
├── requirements.txt
├── INTERVIEW_NOTES.md
└── README.md
```

The model file and metrics are generated locally by `train_model.py` and are intentionally excluded from Git.

## How it works
1. Load the CSV dataset.
2. Separate the input features (`X`) from the target (`y`).
3. Split the data into 80% training and 20% testing sets.
4. Fit preprocessing and Linear Regression on the training set.
5. Evaluate on held-out test data.
6. Save the trained pipeline for the Streamlit app.

## Metrics
- **MAE:** average absolute prediction error in score points.
- **RMSE:** similar to MAE, but penalizes larger errors more strongly.
- **R²:** compares the model with a baseline that predicts the test-set mean; it can be negative when the model performs poorly.

## Limitations and next steps
The dataset is synthetic and may encode artificial relationships, so its metrics do not show real-world predictive accuracy. Next steps could include comparing Ridge Regression and Random Forest, adding cross-validation, inspecting feature importance, and evaluating on a documented public dataset.

## Interview talking points
- **Why train/test split?** To estimate performance on data not used during fitting.
- **Why Pipeline?** It keeps preprocessing and modeling together and helps avoid data leakage.
- **Why MAE?** It gives a simple average error in score points.
- **Why synthetic data is a limitation?** It may not represent real student behavior.
