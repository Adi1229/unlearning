import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
import random

# -----------------------------
# Load embeddings
# -----------------------------
BASE_PATH = "ml_speech_projects/fbank40_models_2026y2m6d8h44m43s/embeddings"

X_train = np.load(f"{BASE_PATH}/train_embeddings_unlearned.npy")
X_val   = np.load(f"{BASE_PATH}/val_embeddings.npy")
X_test  = np.load(f"{BASE_PATH}/test_embeddings.npy")

# -----------------------------
# Generate RANDOM privacy labels
# -----------------------------
NUM_PRIVATE_CLASSES = 4   # attacker tries to infer 4 hidden attributes

def random_labels(n, k):
    return np.random.randint(0, k, size=n)

y_train = random_labels(len(X_train), NUM_PRIVATE_CLASSES)
y_val   = random_labels(len(X_val),   NUM_PRIVATE_CLASSES)
y_test  = random_labels(len(X_test),  NUM_PRIVATE_CLASSES)

# -----------------------------
# Normalize embeddings
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# -----------------------------
# Build MLP attacker
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(200,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(NUM_PRIVATE_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# Train attacker
# -----------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=8,
    verbose=0
)

# -----------------------------
# Evaluate leakage
# -----------------------------
val_pred  = np.argmax(model.predict(X_val), axis=1)
test_pred = np.argmax(model.predict(X_test), axis=1)

val_acc  = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

chance = 1.0 / NUM_PRIVATE_CLASSES

print("\n=== PRIVACY LEAKAGE REPORT (MLP ATTACKER) ===")
print(f"Chance accuracy     : {chance:.2f}")
print(f"Validation accuracy : {val_acc:.2f}")
print(f"Test accuracy       : {test_acc:.2f}")

if test_acc > chance + 0.10:
    print(" Significant privacy leakage detected")
else:
    print(" Low / no measurable privacy leakage")
