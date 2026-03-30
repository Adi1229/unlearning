import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE = "ml_speech_projects/fbank40_models_2026y2m6d8h44m43s/embeddings"

X_train = np.load(f"{BASE}/train_embeddings_unlearned.npy")
y_train = np.load(f"{BASE}/train_labels.npy")

X_val = np.load(f"{BASE}/val_embeddings.npy")
y_val = np.load(f"{BASE}/val_labels.npy")

X_test = np.load(f"{BASE}/test_embeddings.npy")
y_test = np.load(f"{BASE}/test_labels.npy")

clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

clf.fit(X_train, y_train)

print("\nValidation accuracy:")
print(accuracy_score(y_val, clf.predict(X_val)))

print("\nTest accuracy:")
print(accuracy_score(y_test, clf.predict(X_test)))

print("\nDetailed report:")
print(classification_report(y_test, clf.predict(X_test)))
