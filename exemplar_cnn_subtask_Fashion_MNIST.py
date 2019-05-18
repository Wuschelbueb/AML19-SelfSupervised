from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn, test_classification_on_exemplar_cnn
from utils import plot_n_curves, plot_summary

################################################
#     Exemplar CNN sub task Fashion MNIST      #
################################################


def exemplar_cnn_subtask_fashion_mnist():
    # train exemplar cnn with FashionMNIST
    ex_cnn_f_trained, train_losses_ex_cnn_f, train_acc_ex_cnn_f = train_exemplar_cnn()
    plot_n_curves([train_losses_ex_cnn_f], ["train loss"], "Loss train ExemplarCNN FashionMNIST", axis2="Loss")
    plot_n_curves([train_acc_ex_cnn_f], ["train accuracy"], "Accuracy train ExemplarCNN FashionMNIST", axis2="Accuracy")

    # fine tune exemplar cnn with FashionMNIST
    ex_cnn_f_trained_finetuned, train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft, train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft = fine_tune_exemplar_cnn(ex_cnn_f_trained)
    plot_n_curves([train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft], ["train loss", "val loss"], "Loss fine tune ExemplarCNN FashionMNIST", axis2="Loss")
    plot_n_curves([train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune ExemplarCNN FashionMNIST", axis2="Accuracy")

    # test with FashionMNIST
    average_test_loss_ex_cnn_f, average_test_accuracy_ex_cnn_f = test_classification_on_exemplar_cnn(ex_cnn_f_trained_finetuned)
    plot_summary([train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft], average_test_accuracy_ex_cnn_f, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test ExemplarCNN Fashion MNIST", axis2="Accuracy")
    plot_summary([train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft], average_test_loss_ex_cnn_f, ["train loss", "val loss", "test loss"], "Loss Test ExemplarCNN Fashion MNIST", axis2="Loss")

    return average_test_loss_ex_cnn_f, average_test_accuracy_ex_cnn_f
