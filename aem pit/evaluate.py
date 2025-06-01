import matplotlib.pyplot as plt

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
