from pathlib import Path

import pandas as pd
from model.predict import make_prediction

TEST_DATA_PATH = Path(__file__).resolve().parent / "bankchurn_test.csv"

sample_input_data = pd.read_csv(TEST_DATA_PATH)

result = make_prediction(input_data=sample_input_data)

predictions = result.get("predictions")
errors = result.get("errors")

if errors is not None:
    raise ValueError(f"Validation errors: {errors}")

if predictions is None or len(predictions) != len(sample_input_data):
    raise RuntimeError("Prediction count does not match input records.")

print(f"Model version: {result.get('version')}")
print(f"Input records: {len(sample_input_data)}")
print(f"Predictions generated: {len(predictions)}")
print(f"Validation errors: {errors}")
print(f"First 10 predictions: {predictions[:10]}")
print(
    "Prediction distribution:",
    pd.Series(predictions).value_counts().sort_index().to_dict()
)
