# Took help from AI to generate this script.
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
for folder in ["results", "figures"]:
    (ROOT / folder).mkdir(exist_ok=True)
data = pd.read_csv(ROOT / "data/clean.csv")
cutoff = "2012-08-01"
train = data[data.dteday < cutoff]
test = data[data.dteday >= cutoff]
categorical = ["hr", "workingday", "season", "weathersit", "mnth", "weekday", "holiday"]
features = categorical + ["temp", "hum", "windspeed", "yr", "atemp"]
x_train, x_test = train[features], test[features]
y_train, y_test = train.cnt, test.cnt

def encode_categories(columns=categorical):
    return ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore", drop="first"), columns)],
        remainder="passthrough",
    )

models = {
    "Training mean": DummyRegressor(),
    "Linear regression": make_pipeline(encode_categories(), LinearRegression()),
    "Random forest": make_pipeline(
        encode_categories(),
        RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1),
    ),
}
scores = []
for name, model in models.items():
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    scores.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R\u00b2": r2_score(y_test, predictions),
    })
pd.DataFrame(scores).to_csv(
    ROOT / "results/model_scores.csv", index=False, encoding="utf-8-sig"
)

print("Test comparison (August-December 2012):")
print(pd.DataFrame(scores).to_string(index=False))
forest = models["Random forest"]
predictions = forest.predict(x_test)
pd.DataFrame({"dteday": test.dteday, "hr": test.hr, "actual_cnt": y_test, "predicted_cnt": predictions}).to_csv(ROOT / "results/test_predictions.csv", index=False)
importance = permutation_importance(
    forest, x_test, y_test, scoring="neg_mean_absolute_error", n_repeats=8, random_state=0
)
ranking = pd.Series(importance.importances_mean, index=features).sort_values()
ranking.rename("MAE increase").to_csv(ROOT / "results/feature_importance.csv")
high = y_test >= y_test.quantile(.9)
summary = {
    "features": features,
    "target": data.cnt.describe().to_dict(),
    "train_rows": len(train),
    "test_rows": len(test),
    "train_mean": y_train.mean(),
    "test_mean": y_test.mean(),
    "train_start": train.dteday.min(),
    "train_end": train.dteday.max(),
    "test_start": test.dteday.min(),
    "test_end": test.dteday.max(),
    "peak_cutoff": y_test.quantile(.9),
    "peak_bias": (predictions[high] - y_test[high]).mean(),
}
(ROOT / "results/data_summary.json").write_text(json.dumps(summary, indent=2))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
limit = max(y_test.max(), predictions.max()) * 1.04
ax.scatter(y_test, predictions, s=9, alpha=.22, color="#0072B2")
ax.plot([0, limit], [0, limit], "--", color="#444444", label="Perfect prediction")
ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Actual rentals per hour",
       ylabel="Predicted rentals per hour", title="Random forest predictions on held-out hours")
ax.legend(frameon=False)
fig.savefig(ROOT / "figures/fig1_predicted_vs_actual.png", dpi=200)
plt.close(fig)

labels = {"hr": "Hour of day", "workingday": "Working day", "season": "Season",
          "weathersit": "Weather category", "temp": "Temperature", "hum": "Humidity",
          "windspeed": "Wind speed", "yr": "Year", "mnth": "Month",
          "weekday": "Day of week", "holiday": "Holiday", "atemp": "Feels-like temperature"}
fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
ax.barh([labels[name] for name in ranking.index], ranking.values, color="#0072B2")
ax.axvline(0, color="#555555", linewidth=.7)
ax.set(xlabel="Increase in test MAE after shuffling (rentals/hour)",
       title="Random forest feature importance")
fig.savefig(ROOT / "figures/fig2_feature_importance.png", dpi=200)
plt.close(fig)
