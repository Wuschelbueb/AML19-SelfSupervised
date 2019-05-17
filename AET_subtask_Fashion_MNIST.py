from AET import train_aet_mnist_cnn, transfer_learning_aet, test_aet
from utils import plot_n_curves, plot_summary

###############################################
#        AET sub task Fashion MNIST           #
###############################################


def aet_subtask_fashion_MNIST():
    # train AET with FashionMNIST
    aet_f_trained, train_losses_aet_f = train_aet_mnist_cnn()
    plot_n_curves(train_losses_aet_f, "Train losses", "Loss training AET Fashion MNIST", axis2="Loss")

    # transfer learning with FashionMNIST
    encoder, train_losses_aet_f_ft, val_losses_aet_f_ft, train_accuracies_aet_f_ft, val_accuracies_aet_f_ft = transfer_learning_aet(aet_f_trained)
    #plot_n_curves([train_losses_aet, val_losses_aet], ["train loss", "val loss"], "Loss AET FashionMNIST", axis2="Loss")
    #plot_n_curves([train_accuracies_aet, val_accuracies_aet], ["train accuracy", "val accuracy"], "Accuracy AET FashionMNIST", axis2="Accuracy")

    # test with FashionMNIST
    average_test_loss_ae_mnist, average_test_accuracy_ae_mnist = test_aet(encoder)
    plot_summary([train_accuracies_aet_f_ft, val_accuracies_aet_f_ft], average_test_accuracy_ae_mnist, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy AET FashionMNIST", axis2="Accuracy")
    plot_summary([train_losses_aet_f_ft, val_losses_aet_f_ft], average_test_loss_ae_mnist, ["train loss", "val loss", "test average loss"], "Loss AET FashionMNIST", axis2="Loss")