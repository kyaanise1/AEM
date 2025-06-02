import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_costs(cost_logs):
    plt.figure(figsize=(10, 6))
    for label, costs in cost_logs.items():
        plt.plot(costs, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost (MSE)')
    plt.title('Training Cost Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cost_comparison.png')
    plt.show()

def show_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        images, labels = next(iter(test_loader))
        images = images[:num_samples].to(device)
        labels = labels[:num_samples]
        
        # Get predictions
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Create a figure to display images and predictions
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.title(f'Pred: {predicted[i]}\nTrue: {labels[i]}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png')
        plt.show()
