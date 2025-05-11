import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from simplecnn import SimpleCNN
from drivingdataset import DrivingDataset

# === Hyperparameters ===
EPOCHS = 200  # Total epochs
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 5  # Stop if validation loss doesn't improve for 5 epochs

# === Dataset and Dataloader ===
dataset = DrivingDataset("training_data", "training_data/log.json")
train_size = int(0.8 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model, Loss, Optimizer ===
model = SimpleCNN()
model.to(DEVICE)

for param in model.parameters():
    print(param)

criterion = nn.BCEWithLogitsLoss()  # Use logits + sigmoid internally
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(list(model.parameters()), lr=LEARNING_RATE)

# === Early Stopping Variables ===
best_val_loss = float('inf')
epochs_without_improvement = 0

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)  # shape: [batch_size, 4]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)

    # === Validation Loop ===
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{EPOCHS} - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Validation Loss: {avg_val_loss:.4f}")

    # === Early Stopping Logic ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "model_behavioral_cloning_bestv2.pth")
        print("✅ Validation loss improved, model saved.")
    else:
        epochs_without_improvement += 1
        print(f"⚠️ No improvement in validation loss for {epochs_without_improvement} epochs.")
        
    if epochs_without_improvement >= PATIENCE:
        print("⏸️ Early stopping triggered.")
        break

# === Final model saving (if early stopping didn't trigger) ===
torch.save(model.state_dict(), "model_behavioral_cloning_finalv2.pth")
print("✅ Final model saved to model_behavioral_cloning_finalv2.pth")

