import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load CSV
df = pd.read_csv("test_election_news.csv")

y_true = df["label"]
y_pred = df["predicted_label"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))