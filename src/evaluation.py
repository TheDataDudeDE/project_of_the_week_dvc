import pickle
import yaml
from dvclive import Live
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

# Initialize dvclive
live = Live("eval")

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load model and data.
with open(params["model_path"], "rb") as fd:
    model = pickle.load(fd)
print("Model loaded successfully:", model)

# read test data
df_test = pd.read_csv(params["test_data_path"])

# 'species' is the target variable
X_test = df_test.drop("species", axis=1)
y_test = df_test["species"]

# Predicting using the model
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Log metrics

live.log_metric("acc", accuracy)

fig, axes = plt.subplots(dpi=100)
fig.subplots_adjust(bottom=0.2, top=0.95)
axes.set_ylabel("Mean decrease in impurity")
importances = model.feature_importances_
forest_importances = pd.Series(importances, index=X_test.columns).nlargest(n=30)
forest_importances.plot.bar(ax=axes)
live.log_image("importance.png", fig)

# Save metrics and plots to files
live.next_step()
