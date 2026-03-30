import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.manifold import TSNE  # New tool for 2D visualization

# =========================
# Paths
# =========================
BASE_PATH = "ml_speech_projects/fbank40_models_2026y2m6d8h44m43s/embeddings"
# Ensure the directory exists for saving plots
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# Load your data

X_train = np.load(os.path.join(BASE_PATH, "train_embeddings.npy"))
y_train = np.load(os.path.join(BASE_PATH, "train_labels.npy"))
print("Train embeddings shape:", X_train.shape)

# =========================
# Hyperparameters
# =========================
EMB_DIM = X_train.shape[1] 
BATCH_SIZE = 16
EPOCHS = 100       # Set to 100 for a quicker demonstration
LAMBDA_PRIVACY = 1.5 # Increased slightly to make unlearning more visible
LR = 1e-3

# =========================
# Models
# =========================
# The Filter
projection = layers.Dense(EMB_DIM, use_bias=False, name="linear_projection")

# The Good Actor (Task)
task_head = models.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

# The Attacker (Privacy)
privacy_head = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
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
dataset = dataset.shuffle(128).batch(BATCH_SIZE)

# =========================
# Training Loop with Visualization
# =========================
task_losses = []
privacy_losses = []

print("\nStarting gradient ascent unlearning with visual tracking...\n")

for epoch in range(1, EPOCHS + 1):
    epoch_task_loss = 0.0
    epoch_priv_loss = 0.0
    batches = 0
    
    for x, y in dataset:
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            z = projection(x, training=True)
            task_preds = task_head(z, training=True)
            priv_preds = privacy_head(z, training=True)
            
            # 2. Calculate Losses
            task_loss = task_loss_fn(y, task_preds)
            
            # Create random targets to confuse the privacy head
            random_priv_labels = tf.random.uniform(
                shape=(tf.shape(y)[0],), minval=0, maxval=4, dtype=tf.int32
            )
            privacy_loss = privacy_loss_fn(random_priv_labels, priv_preds)
            
            # 3. The "Unlearning" Equation: Minimize Task, Maximize Privacy Error
            total_loss = task_loss - LAMBDA_PRIVACY * privacy_loss
            
        # 4. Apply Gradients (The "Adjustment")
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

    # =========================
    # Visualizing Every 10 Epochs
    # =========================
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Task Loss: {task_losses[-1]:.4f} | Privacy Loss: {privacy_losses[-1]:.4f}")
        
        # Get current state of embeddings
        z_current = projection(X_train, training=False).numpy()
        
        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        z_2d = tsne.fit_transform(z_current)
        
        # Create the Plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_train, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Task Labels (Diagnosis)')
        plt.title(f"Embedding Space at Epoch {epoch}\n(Separation = Task Learning | Mixing = Unlearning)")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        
        # Save each plot to see the progression later
        plt.savefig(os.path.join(BASE_PATH, f"embedding_plot_epoch_{epoch}.png"))
        plt.show() # This will pop up the window during training

print("\nUnlearning complete.")