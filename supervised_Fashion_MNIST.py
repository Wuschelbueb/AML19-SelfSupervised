from supervised import train_supervised_FashionMNIST, test_classification_on_supervised_fashionMNIST
from utils import plot_summary

################################################
#       Supervised subtask Fashion MNIST       #
################################################


def supervised_fashion_mnist():
    # supervised training with Fashion MNIST
    sv_mnist_trained, train_losses_sv_mnist, val_losses_sv_mnist, train_acc_sv_mnist, val_acc_sv_mnist = train_supervised_FashionMNIST()
    # plot_n_curves([train_losses_sv_mnist, val_losses_sv_mnist], ["train loss", "val loss"], "Loss train Supervised Fashion MNIST", axis2="Loss")
    # plot_n_curves([train_acc_sv_mnist, val_acc_sv_mnist], ["train accuracy", "val accuracy"], "Accuracy train supervised Fashion MNIST", axis2="Accuracy")

    # test with Fashion MNIST
    average_test_loss_sv_mnist, average_test_accuracy_sv_mnist = test_classification_on_supervised_fashionMNIST(sv_mnist_trained)
    plot_summary([train_acc_sv_mnist, val_acc_sv_mnist], average_test_accuracy_sv_mnist, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test Supervised Fashion MNIST", axis2="Accuracy")
    plot_summary([train_losses_sv_mnist, val_losses_sv_mnist], average_test_loss_sv_mnist, ["train loss", "val loss", "test loss"], "Loss Test Supervised Fashion MNIST", axis2="Loss")
