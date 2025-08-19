import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import resample

# ------------------------------
# 1. Load Dataset
# ------------------------------
print("📂 Loading dataset...")
data = pd.read_csv("fake_news_dataset.csv")

# Check dataset structure
print("\nDataset Info:")
print(data.info())

# ------------------------------
# 2. Handle Labels Automatically
# ------------------------------
# If labels are already numeric (0 & 1), keep them.
# Otherwise, convert FAKE → 0, REAL → 1
if data['label'].dtype == 'object':
    data['label'] = data['label'].map({'FAKE': 0, 'REAL': 1})

# Validate labels
if set(data['label'].unique()) - {0, 1}:
    raise ValueError("❌ Invalid labels found! Please use either 0/1 or FAKE/REAL.")

# ------------------------------
# 3. Handle Imbalanced Dataset
# ------------------------------
print("\n⚖️ Checking dataset balance...")
print(data['label'].value_counts())

majority = data[data['label'] == 1]  # REAL
minority = data[data['label'] == 0]  # FAKE

if len(minority) / len(majority) < 0.4:
    print("⚠️ Imbalanced dataset detected! Applying upsampling...")
    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42
    )
    data = pd.concat([majority, minority_upsampled])
    print("✅ Dataset balanced!")

print("\nNew Label Distribution:")
print(data['label'].value_counts())

# ------------------------------
# 4. Split Dataset (Stratified)
# ------------------------------
print("\n✂️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

# ------------------------------
# 5. Build Pipeline
# ------------------------------
print("\n⚡ Training model...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('clf', LogisticRegression(max_iter=300, class_weight='balanced'))
])

pipeline.fit(X_train, y_train)

# ------------------------------
# 6. Evaluate Model
# ------------------------------
print("\n📊 Evaluating model...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# ------------------------------
# 7. Save Model
# ------------------------------
joblib.dump(pipeline, "fake_news_pipeline.joblib")
print("\n✅ Model saved as fake_news_pipeline.joblib")
