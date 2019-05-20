from supervised import train_supervised_FashionMNIST, test_classification_on_supervised_fashionMNIST
from utils import plot_summary


################################################
#       Supervised subtask Fashion MNIST       #
################################################

def supervised_fashion_mnist():
    # supervised training with Fashion MNIST
    sv_trained, train_losses, val_losses, train_acc, val_acc = train_supervised_FashionMNIST()

    # test with Fashion MNIST
    average_test_loss, average_test_accuracy = test_classification_on_supervised_fashionMNIST(sv_trained)
    plot_summary([train_acc, val_acc], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy Test Supervised Fashion MNIST", axis2="Accuracy")
    plot_summary([train_losses, val_losses], average_test_loss, ["train loss", "val loss", "test loss"],
                 "Loss Test Supervised Fashion MNIST", axis2="Loss")

    return average_test_loss, average_test_accuracy
