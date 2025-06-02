print("main.py is running")


import torch
import torch.optim as optim
from model import FCN
from train import train_model
from data import get_mnist_loaders
from optimizers.ag_sgd import AGSGDOptimizer
from optimizers.nonlinear_ag_sgd import NonLinearAGSGDOptimizer
from evaluate import plot_costs, show_predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    epochs = 50
    criterion = torch.nn.CrossEntropyLoss()

    optimizers_to_run = {
        #'Vanilla': lambda params: optim.SGD(params, lr=1),
        #'AdaGrad': lambda params: optim.Adagrad(params, lr=1.0),
        #'AdaDelta': lambda params: optim.Adadelta(params, lr=1.0),
        #'AG-SGD': lambda params: AGSGDOptimizer(params, s=1, d=0.90, iter_freq=10),
        'NonLinear-AGSGD': lambda params: NonLinearAGSGDOptimizer(params, s=1, d=0.999, iter_freq=10, k=20.0),  # More aggressive decay
    }

    all_costs = {}
    best_model = None
    best_cost = float('inf')

    for name, opt_fn in optimizers_to_run.items():
        print(f"\n=== Training with {name} ===")
        model = FCN()
        optimizer = opt_fn(model.parameters())
        costs = train_model(model, optimizer, train_loader, criterion, device, epochs)
        all_costs[name] = costs
        
        # Keep track of the best model
        if costs[-1] < best_cost:
            best_cost = costs[-1]
            best_model = model

    plot_costs(all_costs)
    
    # Show predictions using the best model
    print("\n=== Showing Predictions ===")
    show_predictions(best_model, test_loader, device)

# 👇 This is the key part — without this nothing runs
if __name__ == "__main__":
    main()
