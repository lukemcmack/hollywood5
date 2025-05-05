import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

df = pd.read_csv("df_2021test.csv")
df_train = df[df["Set"] == "Train"]
df_test = df[df["Set"] == "Test"]


X_train = df_train.iloc[:, 9:]
X_test = df_test.iloc[:, 9:]
y_train = df_train.loc[:, "Won"]
y_test = df_test.loc[:, "Won"]


X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
X_test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32).unsqueeze(1)


class BinaryTextClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryTextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


pos_weight = (y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()
model = BinaryTextClassifier(input_size=X_train.shape[1])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")


model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    print(predictions)
    preds = (predictions >= max(predictions)).int()
    accuracy = (preds == y_test_tensor.int()).float().mean()
    y_true = y_test_tensor.int().numpy()
    y_pred = preds.numpy()
    precision = precision_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy.item():.2f}")
    print(f"Precision: {precision:.4f}")


class_weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
