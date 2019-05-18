from rotation import train_rotation_net_deep_fashion, fine_tune_rotation_model_deep_fashion, \
    test_classification_on_rotation_model_deep_fashion
from utils import plot_n_curves, plot_summary


###############################################
#     Rotation sub task DeepFashion           #
###############################################


def rotation_subtask_deep_fashion():
    # train rotation net with DeepFashion
    rot_df_trained, train_losses_rot_df, val_losses_rot_df, train_acc_rot_df, val_acc_rot_df = train_rotation_net_deep_fashion()
    plot_n_curves([train_losses_rot_df, val_losses_rot_df], ["train loss", "val loss"], "Loss rotation DeepFashion", axis2="Loss")
    plot_n_curves([train_acc_rot_df, val_acc_rot_df], ["train accuracy", "val accuracy"], "Accuracy rotation DeepFashion", axis2="Accuracy")

    # fine tune rotation net with DeepFashion
    rot_df_finetuned, train_losses_rot_df_ft, val_losses_rot_df_ft, train_acc_rot_df_ft, val_acc_rot_df_ft = fine_tune_rotation_model_deep_fashion(rot_df_trained)
    # plot_n_curves([train_losses_rot_df_ft, val_losses_rot_df_ft], ["train loss", "val loss"], "Loss fine tune rotation DeepFashion", axis2="Loss")
    # plot_n_curves([train_acc_rot_df_ft, val_acc_rot_df_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune rotation DeepFashion", axis2="Accuracy")

    # test with DeepFashion
    average_test_loss_rot_df, average_test_accuracy_rot_df = test_classification_on_rotation_model_deep_fashion(rot_df_finetuned)
    plot_summary([train_acc_rot_df_ft, val_acc_rot_df_ft], average_test_accuracy_rot_df, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test Rotation DeepFashion", axis2="Accuracy")
    plot_summary([train_losses_rot_df_ft, val_losses_rot_df_ft], average_test_loss_rot_df, ["train loss", "val loss", "test loss"], "Loss Test Rotation DeepFashion", axis2="Loss")

    return average_test_loss_rot_df, average_test_accuracy_rot_df
