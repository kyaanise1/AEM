import torch
from torch import nn
from tqdm import tqdm
from utils import to_one_hot

def train_model(model, optimizer, train_loader, criterion, device, epochs=50):
    model.to(device)
    model.train()
    cost_log = []

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            labels = to_one_hot(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(lambda: criterion(model(images), labels)) if hasattr(optimizer, 'prev_grads') else optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        cost_log.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return cost_log
