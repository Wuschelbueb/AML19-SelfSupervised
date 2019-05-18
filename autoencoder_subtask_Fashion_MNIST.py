from autoencoder import train_autoencoder_mnist, transfer_learning_autoencoder_mnist, test_autoencoder_mnist
from utils import plot_n_curves, plot_summary


###############################################
#        AET sub task Fashion MNIST           #
###############################################


def autoencoder_subtask_fashion_MNIST():
    # train autoencoder with FashionMNIST
    encoder_trained, train_losses = train_autoencoder_mnist()
    plot_n_curves(train_losses, "Train losses", "Loss train autoencoder Fashion MNIST", axis2="Loss")

    # transfer learning with FashionMNIST
    encoder_ft, train_losses_ft, val_losses_ft, train_acc_ft, val_acc_ft = transfer_learning_autoencoder_mnist(encoder_trained)
    # plot_n_curves([train_losses_ft, val_losses_ft], ["train loss", "val loss"],
    #               "Loss autoencoder FashionMNIST", axis2="Loss")
    # plot_n_curves([train_acc_ft, val_acc_ft], ["train accuracy", "val accuracy"],
    #               "Accuracy autoencoder FashionMNIST", axis2="Accuracy")

    # test with FashionMNIST
    average_test_loss, average_test_accuracy = test_autoencoder_mnist(encoder_ft)
    plot_summary([train_acc_ft, val_acc_ft], average_test_accuracy,
                 ["train accuracy", "val accuracy", "test accuracy"], "Accuracy autoencoder FashionMNIST",
                 axis2="Accuracy")
    plot_summary([train_losses_ft, val_losses_ft], average_test_loss,
                 ["train loss", "val loss", "test average loss"], "Loss autoencoder FashionMNIST", axis2="Loss")
    return average_test_loss, average_test_accuracy
