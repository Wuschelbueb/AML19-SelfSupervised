from AET import train_aet_deep_fashion_cnn, transfer_learning_aet_deep_fashion, test_aet_deep_fashion
from utils import plot_n_curves, plot_summary

################################################
#      AET sub task Fashion DeepFashion        #
################################################


def aet_subtask_deep_fashion():
    # train AET with DeepFashion
    aet_df_trained, train_losses_aet_df = train_aet_deep_fashion_cnn()
    plot_n_curves(train_losses_aet_df, "Train losses", "Loss training AET DeepFashion", axis2="Loss")

    # transfer learning with DeepFashion
    encoder_df, train_losses_aet_df_ft, val_losses_aet_df_ft, train_accuracies_aet_df_ft, val_accuracies_aet_df_ft = transfer_learning_aet_deep_fashion(aet_df_trained)
    # plot_n_curves([train_losses_aet_df, val_losses_aet_df], ["train loss", "val loss"], "Loss AET FashionMNIST", axis2="Loss")
    # plot_n_curves([train_accuracies_aet_df, val_accuracies_aet_df], ["train accuracy", "val accuracy"], "Accuracy AET DeepFashion", axis2="Accuracy")

    # test with DeepFashion
    average_test_loss_ae_df, average_test_accuracy_ae_df = test_aet_deep_fashion(encoder_df)
    plot_summary([train_accuracies_aet_df_ft, val_accuracies_aet_df_ft], average_test_accuracy_ae_df, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test AET DeepFashion", axis2="Accuracy") # TODO: check if it works
    plot_summary([train_losses_aet_df_ft, val_losses_aet_df_ft], average_test_loss_ae_df, ["train loss", "val loss", "test loss"], "Loss Test AET DeepFashion", axis2="Loss") # TODO: check if it works
