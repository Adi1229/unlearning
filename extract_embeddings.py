import os
import numpy as np
from tensorflow.keras.models import load_model, Model

# -----------------------------
# CONFIG (edit only this block)
# -----------------------------
PROJECT_FOLDER = "fbank40_models_2026y2m6d8h44m43s"
BASE_PATH = "./ml_speech_projects/" + PROJECT_FOLDER

MODEL_PATH = BASE_PATH + "/models/CNNLSTM_speech_commands001.h5"

TRAIN_FEATS = BASE_PATH + "/data_train/train_features.npy"
VAL_FEATS   = BASE_PATH + "/data_val/val_features.npy"
TEST_FEATS  = BASE_PATH + "/data_test/test_features.npy"

EMB_SAVE_PATH = BASE_PATH + "/embeddings"
os.makedirs(EMB_SAVE_PATH, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading trained model...")
model = load_model(MODEL_PATH)

# -----------------------------
# BUILD EMBEDDING MODEL
# -----------------------------
# The Flatten layer outputs 200-D embeddings
# -----------------------------
# FORCE BUILD MODEL GRAPH
# -----------------------------
print("Building model graph with dummy input...")

dummy_input = np.zeros((1, 5, 11, 40, 1), dtype=np.float32)
_ = model(dummy_input)

# -----------------------------
# BUILD EMBEDDING MODEL
# -----------------------------
embedding_layer = model.get_layer("flatten_1").output
embedding_model = Model(
    inputs=model.inputs,
    outputs=embedding_layer
)

print("Embedding model ready.")
embedding_model.summary()


# -----------------------------
# DATA LOADER FUNCTION
# -----------------------------
def load_and_reshape(feat_path):
    data = np.load(feat_path)

    X = data[:, :-1]   # features
    y = data[:, -1]    # labels (repeated per frame)

    timesteps = 5
    frame_width = 11
    frames_per_sample = timesteps * frame_width

    num_samples = X.shape[0] // frames_per_sample

    # Trim to exact multiple
    X = X[:num_samples * frames_per_sample]
    y = y[:num_samples * frames_per_sample]

    # Reshape for CNNLSTM
    X = X.reshape(
        num_samples,
        timesteps,
        frame_width,
        X.shape[1],
        1
    )

    # Take ONE label per sample
    y = y.reshape(num_samples, frames_per_sample)[:, 0]

    return X, y

# -----------------------------
# EXTRACT + SAVE
# -----------------------------
def extract_and_save(split_name, feat_path):
    print(f"\nProcessing {split_name} set...")
    X, y = load_and_reshape(feat_path)

    embeddings = embedding_model.predict(X, verbose=1)

    np.save(f"{EMB_SAVE_PATH}/{split_name}_embeddings.npy", embeddings)
    np.save(f"{EMB_SAVE_PATH}/{split_name}_labels.npy", y)

    print(f"Saved {split_name} embeddings:", embeddings.shape)

# -----------------------------
# RUN
# -----------------------------
extract_and_save("train", TRAIN_FEATS)
extract_and_save("val", VAL_FEATS)
extract_and_save("test", TEST_FEATS)

print("\n Embedding extraction complete.")
