from transformers import pipeline

# Load fake news detection model
classifier = pipeline(
    "text-classification",
    model="jy46604790/Fake-News-Bert-Detect"
)

# Sample text
text = "Election Commission has extended voter registration to January 15."

# Run prediction
result = classifier(text)

# Show result
raw_label = result[0]["label"]
confidence = round(result[0]["score"] * 100, 2)

if raw_label == "LABEL_0":
    label = "FAKE"
else:
    label = "REAL"

print("Prediction:", label)
print("Confidence:", confidence, "%")
