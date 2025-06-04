print("main.py is running")


import torch
import torch.optim as optim
from model import FCN
from train import train_model
from data import get_mnist_loaders
from optimizers.ag_sgd import AGSGDOptimizer
from optimizers.nonlinear_ag_sgd import NonLinearAGSGDOptimizer
from evaluate import plot_costs, plot_metrics, show_predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    epochs = 50
    criterion = torch.nn.CrossEntropyLoss()

    optimizers_to_run = {
        'Vanilla': lambda params: optim.SGD(params, lr=1),
        'AdaGrad': lambda params: optim.Adagrad(params, lr=1.0),
        'AdaDelta': lambda params: optim.Adadelta(params, lr=1.0),
        'AG-SGD': lambda params: AGSGDOptimizer(params, s=1, d=0.999, iter_freq=10),
        'NonLinear-AGSGD': lambda params: NonLinearAGSGDOptimizer(params, s=0.5, d=0.999, iter_freq=10, k=20.0),  # Much slower decay
    }

    all_costs = {}
    best_model = None
    best_cost = float('inf')
    
    # Dictionary to store all models
    all_models = {}
    
    # Dictionary to store final training and test losses
    final_metrics = {
        'train_loss': {},
        'test_loss': {},
        'train_accuracy': {},
        'test_accuracy': {}
    }

    for name, opt_fn in optimizers_to_run.items():
        print(f"\n=== Training with {name} ===")
        model = FCN()
        optimizer = opt_fn(model.parameters())
        costs = train_model(model, optimizer, train_loader, criterion, device, epochs)
        all_costs[name] = costs
        all_models[name] = model  # Store the model
        
        # Evaluate on training set
        model.eval()
        train_loss = 0
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
        
        # Evaluate on test set
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        # Store metrics
        final_metrics['train_loss'][name] = train_loss / len(train_loader)
        final_metrics['test_loss'][name] = test_loss / len(test_loader)
        final_metrics['train_accuracy'][name] = 100. * train_correct / train_total
        final_metrics['test_accuracy'][name] = 100. * test_correct / test_total
        
        # Keep track of the best model
        if test_loss < best_cost:
            best_cost = test_loss
            best_model = model

    # Print final metrics
    print("\n=== Final Results ===")
    for name in optimizers_to_run.keys():
        print(f"\n{name}:")
        print(f"Training Loss: {final_metrics['train_loss'][name]:.4f}")
        print(f"Test Loss: {final_metrics['test_loss'][name]:.4f}")
        print(f"Training Accuracy: {final_metrics['train_accuracy'][name]:.2f}%")
        print(f"Test Accuracy: {final_metrics['test_accuracy'][name]:.2f}%")

    # Plot all metrics
    plot_metrics(final_metrics)
    
    # Plot training costs
    plot_costs(all_costs)
    
    # Show predictions for all models
    print("\n=== Showing Predictions for All Models ===")
    show_predictions(all_models, test_loader, device)

# 👇 This is the key part — without this nothing runs
if __name__ == "__main__":
    main()
