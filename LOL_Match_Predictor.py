import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================
# Data Loading
# =====================
DATASET_PATH = "lol_dataset.csv"
data = pd.read_csv(DATASET_PATH)

X = data.drop('win', axis=1)
y = data['win']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Data Preprocessing
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

print("Data preprocessing complete.")

# =====================
# Logistic Regression Model
# =====================
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(model)

# =====================
# Training Loop
# =====================
epochs = 1000

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# =====================
# Evaluation
# =====================
with torch.no_grad():
    train_acc = ((model(X_train) > 0.5).float() == y_train).float().mean()
    test_acc = ((model(X_test) > 0.5).float() == y_test).float().mean()

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# =====================
# L2 Regularization
# =====================
model_l2 = LogisticRegressionModel(X_train.shape[1])
optimizer_l2 = optim.SGD(
    model_l2.parameters(),
    lr=0.01,
    weight_decay=0.01
)

for epoch in range(1000):
    optimizer_l2.zero_grad()
    loss = criterion(model_l2(X_train), y_train)
    loss.backward()
    optimizer_l2.step()

with torch.no_grad():
    test_acc_l2 = ((model_l2(X_test) > 0.5).float() == y_test).float().mean()

print(f"Test Accuracy with L2 Regularization: {test_acc_l2:.4f}")

# =====================
# Metrics & Visualization
# =====================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

with torch.no_grad():
    y_prob = model_l2(X_test)
    y_pred = (y_prob > 0.5).float()

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

print(classification_report(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.title("ROC Curve")
plt.show()

# =====================
# Save & Load Model
# =====================
torch.save(model_l2.state_dict(), "lol_model.pth")

loaded_model = LogisticRegressionModel(X_train.shape[1])
loaded_model.load_state_dict(torch.load("lol_model.pth"))
loaded_model.eval()

with torch.no_grad():
    loaded_acc = ((loaded_model(X_test) > 0.5).float() == y_test).float().mean()

print(f"Loaded Model Accuracy: {loaded_acc:.4f}")

# =====================
# Learning Rate Comparison
# =====================
learning_rates = [0.01, 0.05, 0.1]
best_lr, best_acc = None, 0

for lr in learning_rates:
    temp_model = LogisticRegressionModel(X_train.shape[1])
    temp_opt = optim.SGD(temp_model.parameters(), lr=lr)

    for _ in range(100):
        temp_opt.zero_grad()
        loss = criterion(temp_model(X_train), y_train)
        loss.backward()
        temp_opt.step()

    with torch.no_grad():
        acc = ((temp_model(X_test) > 0.5).float() == y_test).float().mean()

    print(f"LR {lr}: Accuracy {acc:.4f}")

    if acc > best_acc:
        best_acc, best_lr = acc, lr

print(f"Best Learning Rate: {best_lr}")

# =====================
# Feature Importance
# =====================
weights = model_l2.linear.weight.data.numpy().flatten()

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': weights
}).sort_values(by='Importance', ascending=False)

print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.show()