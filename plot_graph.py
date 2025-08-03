# plot_graph.py
import matplotlib.pyplot as plt
import json

with open("training_log.json", "r") as f:
    log = json.load(f)

plt.plot(log["val_accuracy"], label="Validation Accuracy")
plt.plot(log["train_accuracy"], label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.legend()
plt.show()
