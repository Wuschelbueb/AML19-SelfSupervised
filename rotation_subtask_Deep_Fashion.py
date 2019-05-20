from rotation import train_rotation_net_deep_fashion, fine_tune_rotation_model_deep_fashion, \
    test_classification_on_rotation_model_deep_fashion
from utils import plot_n_curves, plot_summary


###############################################
#     Rotation sub task DeepFashion           #
###############################################

def rotation_subtask_deep_fashion():
    # train rotation net with DeepFashion
    rotnet_trained, train_losses, val_losses, train_acc, val_acc = train_rotation_net_deep_fashion()
    plot_n_curves([train_losses, val_losses], ["train loss", "val loss"], "Loss rotation DeepFashion", axis2="Loss")
    plot_n_curves([train_acc, val_acc], ["train accuracy", "val accuracy"],
                  "Accuracy rotation DeepFashion", axis2="Accuracy")

    # fine tune rotation net with DeepFashion
    rotnet_ft, train_losses_ft, val_losses_ft, train_acc_ft, val_acc_ft = fine_tune_rotation_model_deep_fashion(
        rotnet_trained)

    # test with DeepFashion
    average_test_loss, average_test_accuracy = test_classification_on_rotation_model_deep_fashion(rotnet_ft)
    plot_summary([train_acc_ft, val_acc_ft], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy Test Rotation DeepFashion", axis2="Accuracy")
    plot_summary([train_losses_ft, val_losses_ft], average_test_loss, ["train loss", "val loss", "test loss"],
                 "Loss Test Rotation DeepFashion", axis2="Loss")

    return average_test_loss, average_test_accuracy
