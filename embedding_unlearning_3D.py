import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# =========================
# Paths
# =========================
BASE_PATH = "ml_speech_projects/fbank40_models_2026y2m6d8h44m43s/embeddings"
os.makedirs(BASE_PATH, exist_ok=True)

# =========================
# Load Data
# =========================
X_train = np.load(os.path.join(BASE_PATH, "train_embeddings.npy"))
y_train = np.load(os.path.join(BASE_PATH, "train_labels.npy"))

# =========================
# Tuned Hyperparameters
# =========================
EMB_DIM = X_train.shape[1]
BATCH_SIZE = 32
EPOCHS = 150
LAMBDA_PRIVACY = 2.0
LR = 5e-4

projection = layers.Dense(EMB_DIM, use_bias=False, name="linear_projection")

task_head = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(2, activation="softmax")
])

privacy_head = models.Sequential([
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(4, activation="softmax")
])

optimizer = optimizers.Adam(LR)
task_loss_fn = losses.SparseCategoricalCrossentropy()
privacy_loss_fn = losses.SparseCategoricalCrossentropy()

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(512).batch(BATCH_SIZE)

embedding_snapshots = []
epoch_marks = []

print("\nStarting 3D unlearning...\n")

for epoch in range(1, EPOCHS + 1):
    for x, y in dataset:
        with tf.GradientTape() as tape:
            z = projection(x, training=True)
            task_preds = task_head(z, training=True)
            priv_preds = privacy_head(z, training=True)

            task_loss = task_loss_fn(y, task_preds)

            random_priv_labels = tf.random.uniform(
                shape=(tf.shape(y)[0],), minval=0, maxval=4, dtype=tf.int32
            )
            privacy_loss = privacy_loss_fn(random_priv_labels, priv_preds)

            total_loss = task_loss - LAMBDA_PRIVACY * privacy_loss

        vars_to_train = (projection.trainable_variables +
                         task_head.trainable_variables +
                         privacy_head.trainable_variables)

        grads = tape.gradient(total_loss, vars_to_train)
        optimizer.apply_gradients(zip(grads, vars_to_train))

    if epoch % 10 == 0 or epoch == 1:
        z_current = projection(X_train, training=False).numpy()
        embedding_snapshots.append(z_current)
        epoch_marks.append(epoch)

# =========================
# 3D Visualization
# =========================
print("\nGenerating 3D visualization...")

rows = 3
cols = 5
fig = plt.figure(figsize=(20, 12))

for i, (emb, ep) in enumerate(zip(embedding_snapshots, epoch_marks)):
    ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
    pca = PCA(n_components=3)
    z_3d = pca.fit_transform(emb)

    ax.scatter(z_3d[:, 0], z_3d[:, 1], z_3d[:, 2], c=y_train, cmap='viridis', alpha=0.6)
    ax.set_title(f"Epoch {ep}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "ALL_3D_EPOCH_PROGRESS.png"))
plt.close()

print("\n3D Unlearning complete.")