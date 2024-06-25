import matplotlib.pyplot as plt

def plot_f1_scores(train_f1_scores, dev_f1_scores, test_f1, best_epoch, num_epochs, learn, drop, batch_size, seed):
    # Plotting the F1 scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_f1_scores, label='Train F1 Score')
    plt.plot(range(1, num_epochs + 1), dev_f1_scores, label='Development F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.legend()

    print(f"Test F1 Score of the best model: {test_f1}")

    # Add a single red point for test F1 score
    plt.plot([best_epoch + 1], [test_f1], 'ro', label='Test F1 Score')
    plt.text(best_epoch + 1, test_f1, f'{test_f1:.6f}', fontsize=12, ha='right', color='red')

    plt.legend()

    # Force integer ticks on the x-axis
    plt.xticks(range(1, num_epochs + 1))

    plt.savefig(f'f1_scores_plot_lr={learn}&d={drop}&batch={batch_size}&seed={seed}.png')
    try:
        pass
        #plt.show()
    except Exception as e:
        print(f"Could not display the plot: {e}")
