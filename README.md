# Student Performance Predictor

A Python application that estimates a sample student score from study hours, attendance, previous score, assignment completion, and sleep hours.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell, activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies and train the model:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit will display a local URL to open in your browser.

## Files

- `train_model.py` — trains and evaluates the Linear Regression model.
- `app.py` — provides the interactive interface.
- `data/student_performance.csv` — sample data used for training.
- `requirements.txt` — required Python libraries.

## Model

The training script splits the data into training and test sets, applies median imputation and scaling through a scikit-learn pipeline, and evaluates the model using MAE, RMSE, and R².

The included CSV contains generated sample data, so the results are illustrative rather than validated estimates for real students.
