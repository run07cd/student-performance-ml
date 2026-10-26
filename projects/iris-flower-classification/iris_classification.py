"""Beginner classification example using scikit-learn's built-in Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Iris Flower Classification")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=dataset.target_names))
    example = [[5.1, 3.5, 1.4, 0.2]]
    predicted_class = model.predict(example)[0]
    print("Example measurements (sepal length, sepal width, petal length, petal width):", example[0])
    print("Predicted flower type:", dataset.target_names[predicted_class])

if __name__ == "__main__":
    main()
