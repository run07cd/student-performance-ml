"""Beginner regression example using generated synthetic housing data."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def make_demo_data(rows=400, seed=42):
    """Create artificial data to demonstrate a regression workflow."""
    rng = np.random.default_rng(seed)
    square_feet = rng.integers(500, 3501, size=rows)
    bedrooms = rng.integers(1, 6, size=rows)
    age_years = rng.integers(0, 81, size=rows)
    distance_km = rng.uniform(1, 30, size=rows)
    noise = rng.normal(0, 18000, size=rows)
    price = (
        30000 + square_feet * 180 + bedrooms * 12000
        - age_years * 900 - distance_km * 1500 + noise
    )
    return pd.DataFrame({
        "square_feet": square_feet,
        "bedrooms": bedrooms,
        "age_years": age_years,
        "distance_to_city_km": distance_km.round(1),
        "price": price.round(0),
    })

def main():
    data = make_demo_data()
    features = ["square_feet", "bedrooms", "age_years", "distance_to_city_km"]
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data["price"], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("House Price Prediction (synthetic demo)")
    print(f"Mean Absolute Error (price units): {mean_absolute_error(y_test, predictions):,.0f}")
    print(f"R² score: {r2_score(y_test, predictions):.3f}")
    example = pd.DataFrame([{
        "square_feet": 1500,
        "bedrooms": 3,
        "age_years": 10,
        "distance_to_city_km": 8.0,
    }])
    predicted_price = model.predict(example)[0]
    print(f"Example estimate for a 1,500 sq ft, 3-bedroom demo house: {predicted_price:,.0f} price units")
    print("\nImportant: prices and relationships are generated artificially; these are not real market estimates.")

if __name__ == "__main__":
    main()
