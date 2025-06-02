import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_costs(costs_dict):
    plt.figure(figsize=(10, 6))
    for name, costs in costs_dict.items():
        plt.plot(costs, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_costs.png')
    plt.close()

def plot_metrics(final_metrics):
    metrics = ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']
    titles = ['Training Loss', 'Test Loss', 'Training Accuracy', 'Test Accuracy']
    optimizers = list(final_metrics['train_loss'].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        values = [final_metrics[metric][opt] for opt in optimizers]
        bars = ax.bar(optimizers, values)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_ylabel('Value')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add percentage sign for accuracy metrics
        if 'accuracy' in metric:
            ax.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()

def show_predictions(models_dict, test_loader, device):
    # Get a batch of test data
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Create a figure for each model
    for model_name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # Move tensors to CPU for plotting
            images_cpu = images.cpu()
            predicted_cpu = predicted.cpu()
            labels_cpu = labels.cpu()
            
            # Plot first 10 images with predictions
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(f'Predictions - {model_name}', fontsize=16)
            
            for i in range(10):
                row = i // 5
                col = i % 5
                ax = axes[row, col]
                
                # Display image
                ax.imshow(images_cpu[i].squeeze(), cmap='gray')
                
                # Set title with prediction and true label
                color = 'green' if predicted_cpu[i] == labels_cpu[i] else 'red'
                ax.set_title(f'Pred: {predicted_cpu[i]}\nTrue: {labels_cpu[i]}', color=color)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'predictions_{model_name}.png')
            plt.close()
