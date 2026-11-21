from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "data/hour.csv")
columns = ["dteday", "hr", "workingday", "season", "weathersit",
           "temp", "hum", "windspeed", "yr", "mnth", "weekday", "holiday", "atemp", "cnt"]
data = raw[columns].sort_values(["dteday", "hr"])
assert not data.isna().any().any()
assert not data.duplicated(["dteday", "hr"]).any()
data.to_csv(ROOT / "data/clean.csv", index=False)
