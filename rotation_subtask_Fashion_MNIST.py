from rotation import train_rotation_net, fine_tune_rotation_model, test_classification_on_rotation_model
from utils import plot_n_curves, plot_summary


###############################################
#     Rotation sub task Fashion MNIST         #
###############################################


def rotation_subtask_fashion_mnist():
    #train rotation net with FashionMNIST
    rot_f_trained, train_losses_rot_f, val_losses_rot_f, train_acc_rot_f, val_acc_rot_f = train_rotation_net()
    plot_n_curves([train_losses_rot_f, val_losses_rot_f], ["train loss", "val loss"], "Loss rotation FashionMNIST", axis2="Loss")
    plot_n_curves([train_acc_rot_f, val_acc_rot_f], ["train accuracy", "val accuracy"], "Accuracy rotation FashionMNIST", axis2="Accuracy")

    # fine tune rotation net with FashionMNIST
    rot_f_finetuned, train_losses_rot_f_ft, val_losses_rot_f_ft, train_acc_rot_f_ft, val_acc_rot_f_ft = fine_tune_rotation_model(rot_f_trained)
    # plot_n_curves([train_losses_rot_f_ft, val_losses_rot_f_ft], ["train loss", "val loss"], "Loss fine tune rotation FashionMNIST", axis2="Loss")
    # plot_n_curves([train_acc_rot_f_ft, val_acc_rot_f_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune rotation FashionMNIST", axis2="Accuracy")

    # test with FashionMNIST
    average_test_loss_rot_mnist, average_test_accuracy_rot_mnist = test_classification_on_rotation_model(rot_f_finetuned)
    plot_summary([train_acc_rot_f_ft, val_acc_rot_f_ft], average_test_accuracy_rot_mnist, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test Rotation DeepFashion", axis2="Accuracy") # TODO: check if it works
    plot_summary([train_losses_rot_f_ft, val_losses_rot_f_ft], average_test_loss_rot_mnist, ["train loss", "val loss", "test loss"], "Loss Test Rotation DeepFashion", axis2="Loss") # TODO: check if it works
