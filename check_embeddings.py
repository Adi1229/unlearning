import numpy as np
BASE = "ml_speech_projects/fbank40_models_2026y2m6d8h44m43s/embeddings"

X_train = np.load(f"{BASE}/train_embeddings.npy")
X_val   = np.load(f"{BASE}/val_embeddings.npy")
X_test  = np.load(f"{BASE}/test_embeddings.npy")


print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)
print("NaNs in train:", np.isnan(X_train).any())
