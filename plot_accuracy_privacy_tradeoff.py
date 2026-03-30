import matplotlib.pyplot as plt
import numpy as np

# ======== RESULTS FROM YOUR EXPERIMENTS ========

labels = ["Baseline", "After Unlearning"]
x = np.arange(len(labels))

# Task accuracy
task_accuracy = [0.75, 0.50]

# Privacy leakage – Random Label attacker
privacy_random = [0.25, 0.00]

# Privacy leakage – MLP attacker
privacy_mlp = [0.38, 0.38]

# ======== PLOTTING ========

plt.figure(figsize=(9, 5))

plt.plot(x, task_accuracy, marker='o', linewidth=2.5,
         label="Task Accuracy", color="tab:blue")

plt.plot(x, privacy_random, marker='s', linewidth=2.5,
         label="Privacy Leakage (Random Labels)", color="tab:green")

plt.plot(x, privacy_mlp, marker='^', linewidth=2.5,
         label="Privacy Leakage (MLP Attacker)", color="tab:red")

plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy–Privacy Tradeoff Under Different Attackers")

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig("accuracy_privacy_tradeoff_attackers.png", dpi=300)
plt.show()

print("Plot saved as accuracy_privacy_tradeoff_attackers.png")

