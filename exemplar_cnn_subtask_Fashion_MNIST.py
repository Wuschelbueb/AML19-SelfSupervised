from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn, test_classification_on_exemplar_cnn
from utils import plot_n_curves, plot_summary


################################################
#     Exemplar CNN sub task Fashion MNIST      #
################################################

def exemplar_cnn_subtask_fashion_mnist():
    # train exemplar cnn with FashionMNIST
    ex_cnn_trained, losses, accuracies = train_exemplar_cnn()
    plot_n_curves([losses], ["train loss"], "Loss train ExemplarCNN FashionMNIST", axis2="Loss")
    plot_n_curves([accuracies], ["train accuracy"], "Accuracy train ExemplarCNN FashionMNIST", axis2="Accuracy")

    # fine tune exemplar cnn with FashionMNIST
    ex_cnn_finetuned, train_losses, val_losses, train_acc, val_acc = fine_tune_exemplar_cnn(ex_cnn_trained)

    # test with FashionMNIST
    average_test_loss, average_test_accuracy = test_classification_on_exemplar_cnn(ex_cnn_finetuned)
    plot_summary([train_acc, val_acc], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy Test ExemplarCNN Fashion MNIST", axis2="Accuracy")
    plot_summary([train_losses, val_losses], average_test_loss, ["train loss", "val loss", "test loss"],
                 "Loss Test ExemplarCNN Fashion MNIST", axis2="Loss")

    return average_test_loss, average_test_accuracy
