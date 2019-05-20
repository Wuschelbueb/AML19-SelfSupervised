from supervised import test_classification_deep_fashion, train_supervised_deep_fashion
from utils import plot_summary


################################################
#       Supervised subtask Deep Fashion        #
################################################

def supervised_deep_fashion():
    # supervised training with DeepFashion
    sv_trained, train_losses, val_losses, train_acc, val_acc = train_supervised_deep_fashion()

    # test with DeepFashion
    average_test_loss, average_test_accuracy = test_classification_deep_fashion(sv_trained)
    plot_summary([train_acc, val_acc], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy Test Supervised Deep Fashion", axis2="Accuracy")
    plot_summary([train_losses, val_losses], average_test_loss, ["train loss", "val loss", "test loss"],
                 "Loss Test Supvervised Deep Fashion", axis2="Loss")

    return average_test_loss, average_test_accuracy
