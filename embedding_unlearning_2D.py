import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.manifold import TSNE

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
print("Train embeddings shape:", X_train.shape)

# =========================
# Tuned Hyperparameters
# =========================
EMB_DIM = X_train.shape[1]
BATCH_SIZE = 32              # Increased
EPOCHS = 150                 # Increased
LAMBDA_PRIVACY = 0.5         # Stronger unlearning
LR = 1e-4                    # Lower LR for stability

# =========================
# Models 
# =========================
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

# =========================
# Optimizers & Losses
# =========================
optimizer = optimizers.Adam(LR)
task_loss_fn = losses.SparseCategoricalCrossentropy()
privacy_loss_fn = losses.SparseCategoricalCrossentropy()

# =========================
# Dataset
# =========================
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(512).batch(BATCH_SIZE)

# =========================
# Training Loop
# =========================
task_losses = []
privacy_losses = []
embedding_snapshots = []
epoch_marks = []

print("\nStarting tuned gradient ascent unlearning...\n")

for epoch in range(1, EPOCHS + 1):
    epoch_task_loss = 0.0
    epoch_priv_loss = 0.0
    batches = 0
    
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
        
        epoch_task_loss += task_loss.numpy()
        epoch_priv_loss += privacy_loss.numpy()
        batches += 1

    task_losses.append(epoch_task_loss / batches)
    privacy_losses.append(epoch_priv_loss / batches)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Task Loss: {task_losses[-1]:.4f} | Privacy Loss: {privacy_losses[-1]:.4f}")
        z_current = projection(X_train, training=False).numpy()
        embedding_snapshots.append(z_current)
        epoch_marks.append(epoch)

# =========================
# SINGLE LARGE 2D PLOT
# =========================
print("\nGenerating combined 2D visualization...")

num_plots = len(embedding_snapshots)

cols = 5
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))

# If only one row, make axes iterable
if rows == 1:
    axes = np.expand_dims(axes, axis=0)

axes = axes.flatten()

for i, (emb, ep) in enumerate(zip(embedding_snapshots, epoch_marks)):
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(emb)

    axes[i].scatter(z_2d[:, 0], z_2d[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    axes[i].set_title(f"Epoch {ep}")
    axes[i].set_xticks([])
    axes[i].set_yticks([])

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "ALL_2D_EPOCH_PROGRESS2.png"))
plt.close()

print("\nUnlearning complete.2D visualizations.")