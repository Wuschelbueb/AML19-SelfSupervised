from supervised_deep_fashion import test_classification_deep_fashion, train_supervised_deep_fashion
from utils import plot_summary

################################################
#       Supervised subtask Deep Fashion        #
################################################


def supervised_deep_fashion():
    # supervised training with DeepFashion
    sv_df_trained, train_losses_sv_df, val_losses_sv_df, train_acc_sv_df, val_acc_sv_df = train_supervised_deep_fashion()
    # plot_n_curves([train_losses_sv_df, val_losses_sv_df], ["train loss", "val loss"], "Loss train supervised DeepFashion", axis2="Loss")
    # plot_n_curves([train_acc_sv_df, val_acc_sv_df], ["train accuracy", "val accuracy"], "Accuracy train supervised DeepFashion", axis2="Accuracy")

    # # test with DeepFashion
    average_test_loss_sv_df, average_test_accuracy_sv_df = test_classification_deep_fashion(sv_df_trained)
    plot_summary([train_acc_sv_df, val_acc_sv_df], average_test_accuracy_sv_df, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test Supervised Deep Fashion", axis2="Accuracy") # TODO: check if it works
    plot_summary([train_losses_sv_df, val_losses_sv_df], average_test_loss_sv_df, ["train loss", "val loss", "test loss"], "Loss Test Supvervised Deep Fashion", axis2="Loss") # TODO: check if it works
