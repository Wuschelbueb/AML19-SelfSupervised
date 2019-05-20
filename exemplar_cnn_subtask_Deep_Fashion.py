from exemplar_cnn import test_classification_on_exemplar_cnn_deep_fashion, fine_tune_exemplar_cnn_deep_fashion, \
    train_exemplar_cnn_deep_fashion
from utils import plot_summary, plot_n_curves


################################################
#     Exemplar CNN sub task Deep Fashion       #
################################################

def exemplar_cnn_subtask_deep_fashion():
    # train exemplar cnn with DeepFashion
    ex_cnn_trained, losses, accuracies = train_exemplar_cnn_deep_fashion()
    plot_n_curves([losses], ["train loss"], "Loss train ExemplarCNN DeepFashion", axis2="Loss")
    plot_n_curves([accuracies], ["train accuracy"], "Accuracy train ExemplarCNN DeepFashion", axis2="Accuracy")

    # fine tune exemplar cnn with DeepFashion
    ex_cnn_finetuned, train_losses, val_losses, train_acc, val_acc = fine_tune_exemplar_cnn_deep_fashion(ex_cnn_trained)

    # test with DeepFashion
    average_test_loss, average_test_accuracy = test_classification_on_exemplar_cnn_deep_fashion(ex_cnn_finetuned)
    plot_summary([train_acc, val_acc], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy Test ExemplarCNN Deep Fashion", axis2="Accuracy")
    plot_summary([train_losses, val_losses], average_test_loss, ["train loss", "val loss", "test loss"],
                 "Loss Test ExemplarCNN Deep Fashion", axis2="Loss")

    return average_test_loss, average_test_accuracy
