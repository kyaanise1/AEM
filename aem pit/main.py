print("main.py is running")


import torch
import torch.optim as optim
from model import FCN
from train import train_model
from data import get_mnist_loaders
from optimizers.ag_sgd import AGSGDOptimizer
from optimizers.nonlinear_ag_sgd import NonLinearAGSGDOptimizer
from evaluate import plot_costs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _ = get_mnist_loaders(batch_size=8)

    epochs = 50
    criterion = torch.nn.MSELoss()

    optimizers_to_run = {
        'Vanilla': lambda params: optim.SGD(params, lr=1.0),
        'AdaGrad': lambda params: optim.Adagrad(params, lr=1.0),
        'AdaDelta': lambda params: optim.Adadelta(params, lr=1.0),
        'AG-SGD': lambda params: AGSGDOptimizer(params, s=1.0, d=0.5, iter_freq=5),
        'NonLinear-AGSGD': lambda params: NonLinearAGSGDOptimizer(params, s=1.0, d=0.5, iter_freq=5, k=5.0),
    }

    all_costs = {}

    for name, opt_fn in optimizers_to_run.items():
        print(f"\n=== Training with {name} ===")
        model = FCN()
        optimizer = opt_fn(model.parameters())
        costs = train_model(model, optimizer, train_loader, criterion, device, epochs)
        all_costs[name] = costs

    plot_costs(all_costs)

# 👇 This is the key part — without this nothing runs
if __name__ == "__main__":
    main()
