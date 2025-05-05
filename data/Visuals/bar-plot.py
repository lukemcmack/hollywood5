import matplotlib.pyplot as plt

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")

# Model names and correct prediction counts
models = [
    "Bag-of-Words",
    "Embedding",
    "Temporal-Weighted\nNaive Bayes",
    "Neural Network",
    "Gradient Boosting"
]
correct_predictions = [1, 2, 1, 0, 3]

# Create the bar chart
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, correct_predictions, color='#4C72B0', edgecolor='black')

# Annotate each bar with the count
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1, str(height),
            ha='center', va='bottom', fontsize=10)

# Set chart title and labels
ax.set_title("Correctly Predicted Best Picture Winners by Model", fontsize=16, weight='bold')
ax.set_ylabel("Correct Predictions (out of 11 years)", fontsize=12)
ax.set_ylim(0, 4)

# Remove vertical grid lines
ax.xaxis.grid(False)  # disable vertical lines
ax.yaxis.grid(True, linestyle='--', alpha=0.7)  # keep horizontal lines

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
