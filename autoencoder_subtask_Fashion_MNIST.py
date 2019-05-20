from autoencoder import train_autoencoder_mnist, fine_tune_autoencoder_mnist, test_autoencoder_mnist
from utils import plot_n_curves, plot_summary


###############################################
#        AET sub task Fashion MNIST           #
###############################################

def autoencoder_subtask_fashion_mnist():
    # train autoencoder with FashionMNIST
    encoder_trained, losses = train_autoencoder_mnist()
    plot_n_curves(losses, "Train losses", "Loss train autoencoder Fashion MNIST", axis2="Loss")

    # transfer learning with FashionMNIST
    encoder_ft, train_losses, val_losses, train_acc, val_acc = fine_tune_autoencoder_mnist(encoder_trained)

    # test with FashionMNIST
    average_test_loss, average_test_accuracy = test_autoencoder_mnist(encoder_ft)
    plot_summary([train_acc, val_acc], average_test_accuracy,
                 ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy test autoencoder FashionMNIST", axis2="Accuracy")
    plot_summary([train_losses, val_losses], average_test_loss,
                 ["train loss", "val loss", "test average loss"],
                 "Loss test autoencoder FashionMNIST", axis2="Loss")
    return average_test_loss, average_test_accuracy
