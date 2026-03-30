import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import imageio

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
# STORE INITIAL STATE (Before Unlearning)
# =========================
print("\nSaving initial model state...")

z_initial = projection(X_train, training=False)
task_preds_initial = task_head(z_initial, training=False)
task_classes_initial = tf.argmax(task_preds_initial, axis=1).numpy()

cm_before = confusion_matrix(y_train, task_classes_initial)

plt.figure(figsize=(6,5))
sns.heatmap(cm_before, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix BEFORE Unlearning")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(BASE_PATH, "confusion_matrix_BEFORE4.png"))
plt.close()

# =========================
# Training Loop (UNCHANGED)
# =========================

task_losses = []
privacy_losses = []
embedding_snapshots = []
epoch_marks = []

print("\nStarting gradient ascent unlearning...\n")

# Store epoch 0 snapshot
embedding_snapshots.append(z_initial.numpy())
epoch_marks.append(0)

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

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Task Loss: {task_losses[-1]:.4f} | Privacy Loss: {privacy_losses[-1]:.4f}")
        z_current = projection(X_train, training=False).numpy()
        embedding_snapshots.append(z_current)
        epoch_marks.append(epoch)

# =========================
# CONFUSION MATRIX AFTER
# =========================

z_final = projection(X_train, training=False)
task_preds_final = task_head(z_final, training=False)
task_classes_final = tf.argmax(task_preds_final, axis=1).numpy()

cm_after = confusion_matrix(y_train, task_classes_final)

plt.figure(figsize=(6,5))
sns.heatmap(cm_after, annot=True, fmt='d', cmap="Reds")
plt.title("Confusion Matrix AFTER Unlearning")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(BASE_PATH, "confusion_matrix_AFTER4.png"))
plt.close()

# =========================
# 2D VISUALIZATION GRID
# =========================

print("\nGenerating 2D grid visualization...")

num_plots = len(embedding_snapshots)
cols = 5
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4))
axes = axes.flatten()

image_paths = []

for i, (emb, ep) in enumerate(zip(embedding_snapshots, epoch_marks)):
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(emb)

    axes[i].scatter(z_2d[:, 0], z_2d[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    axes[i].set_title(f"Epoch {ep}")
    axes[i].set_xticks([])
    axes[i].set_yticks([])

    # Save individual frame for GIF
    frame_path = os.path.join(BASE_PATH, f"frame_{ep}.png")
    plt.figure(figsize=(6,5))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    plt.title(f"Epoch {ep}")
    plt.savefig(frame_path)
    plt.close()
    image_paths.append(frame_path)

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
grid_path = os.path.join(BASE_PATH, "ALL_2D_EPOCH_PROGRESS4.png")
plt.savefig(grid_path)
plt.close()

# =========================
# CREATE GIF
# =========================

print("\nCreating animated GIF...")

images = []
for path in image_paths:
    images.append(imageio.imread(path))

gif_path = os.path.join(BASE_PATH, "embedding_progression.gif")
imageio.mimsave(gif_path, images, duration=0.8)

print("\nUnlearning complete.")
print("Saved:")
print("- Confusion Matrix BEFORE")
print("- Confusion Matrix AFTER")
print("- 2D Grid Visualization")
print("- Animated GIF")