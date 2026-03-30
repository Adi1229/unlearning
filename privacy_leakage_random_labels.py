import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load embeddings
# -----------------------------
BASE = "ml_speech_projects/fbank40_models_2026y2m6d8h44m43s/embeddings"

X_train = np.load(f"{BASE}/train_embeddings_unlearned.npy")
X_val   = np.load(f"{BASE}/val_embeddings.npy")
X_test  = np.load(f"{BASE}/test_embeddings.npy")

# -----------------------------
# Create RANDOM privacy labels
# -----------------------------
num_privacy_classes = 4  # attacker assumes 4 hidden attributes

rng = np.random.default_rng(seed=42)
y_train_priv = rng.integers(0, num_privacy_classes, size=len(X_train))
y_val_priv   = rng.integers(0, num_privacy_classes, size=len(X_val))
y_test_priv  = rng.integers(0, num_privacy_classes, size=len(X_test))

chance_level = 1.0 / num_privacy_classes

# -----------------------------
# Normalize embeddings
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# -----------------------------
# Train privacy attacker
# -----------------------------
attacker = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial"
)

attacker.fit(X_train, y_train_priv)

# -----------------------------
# Evaluate leakage
# -----------------------------
val_acc  = accuracy_score(y_val_priv, attacker.predict(X_val))
test_acc = accuracy_score(y_test_priv, attacker.predict(X_test))

print("\n=== PRIVACY LEAKAGE REPORT (RANDOM LABEL ATTACK) ===")
print(f"Chance accuracy        : {chance_level:.2f}")
print(f"Validation accuracy    : {val_acc:.2f}")
print(f"Test accuracy          : {test_acc:.2f}")

if test_acc > chance_level + 0.05:
    print("  Privacy leakage detected")
else:
    print(" Low privacy leakage")
