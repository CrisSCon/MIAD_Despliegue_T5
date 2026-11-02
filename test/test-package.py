from pathlib import Path

import pandas as pd
from model.predict import make_prediction

TEST_DATA_PATH = Path(__file__).resolve().parent / "bankchurn_test.csv"

sample_input_data = pd.read_csv(TEST_DATA_PATH)

result = make_prediction(input_data=sample_input_data)

print(result)