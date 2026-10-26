# Quick Project Explanation for Your Professor

**One-line summary:** I built a small end-to-end regression project that trains a model to estimate a sample score and exposes it through a Streamlit app.

## Explain it in 60 seconds
1. I used Pandas to load a tabular dataset.
2. I selected five input features and one target column.
3. I split the data into training and testing subsets.
4. I used a Scikit-learn Pipeline for median imputation, scaling, and Linear Regression.
5. I evaluated predictions using MAE, RMSE, and R².
6. I built a Streamlit interface to collect inputs and display a prediction.

## Be honest about the limitation
The included data is synthetic, so the results only demonstrate the workflow. Before making real-world claims, I would use a documented public dataset, validate the model carefully, and consider fairness and privacy.

## Questions you should be ready for
- What is supervised learning?
- Why is this a regression task rather than classification?
- What is the difference between training and testing data?
- What do MAE, RMSE, and R² mean?
- What is data leakage, and how does a pipeline help?
- Why can a model trained on synthetic data fail on real data?
- What would you improve next?
