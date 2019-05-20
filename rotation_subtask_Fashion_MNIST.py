from rotation import train_rotation_net, fine_tune_rotation_model, test_classification_on_rotation_model
from utils import plot_n_curves, plot_summary


###############################################
#     Rotation sub task Fashion MNIST         #
###############################################

def rotation_subtask_fashion_mnist():
    # train rotation net with FashionMNIST
    rotnet_trained, train_losses, val_losses, train_acc, val_acc = train_rotation_net()
    plot_n_curves([train_losses, val_losses], ["train loss", "val loss"],
                  "Loss rotation FashionMNIST", axis2="Loss")
    plot_n_curves([train_acc, val_acc], ["train accuracy", "val accuracy"],
                  "Accuracy rotation FashionMNIST", axis2="Accuracy")

    # fine tune rotation net with FashionMNIST
    rotnet_ft, train_losses_ft, val_losses_ft, train_acc_ft, val_acc_ft = fine_tune_rotation_model(rotnet_trained)

    # test with FashionMNIST
    average_test_loss, average_test_accuracy = test_classification_on_rotation_model(rotnet_ft)
    plot_summary([train_acc_ft, val_acc_ft], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy Test Rotation DeepFashion", axis2="Accuracy")
    plot_summary([train_losses_ft, val_losses_ft], average_test_loss, ["train loss", "val loss", "test loss"],
                 "Loss Test Rotation DeepFashion", axis2="Loss")

    return average_test_loss, average_test_accuracy
