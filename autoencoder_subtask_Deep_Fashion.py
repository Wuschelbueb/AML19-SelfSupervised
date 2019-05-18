from autoencoder import train_autoencoder_deep_fashion, transfer_learning_autoencoder_deep_fashion, test_autoencoder_deep_fashion
from utils import plot_n_curves, plot_summary

################################################
#      AET sub task Fashion DeepFashion        #
################################################


def autoencoder_subtask_deep_fashion():
    # train AET with DeepFashion
    encoder_trained, decoder_trained, train_losses = train_autoencoder_deep_fashion()
    plot_n_curves(train_losses, "Train losses", "Loss training AET DeepFashion", axis2="Loss")

    # transfer learning with DeepFashion
    encoder_ft, decoder_ft, classifier_ft, train_losses_ft, val_losses_ft, train_acc_ft, val_acc_ft = transfer_learning_autoencoder_deep_fashion(encoder_trained, decoder_trained)
    plot_n_curves([train_losses_ft, val_losses_ft], ["train loss", "val loss"], "Loss autoencoder DeepFashion", axis2="Loss")
    plot_n_curves([train_acc_ft, val_acc_ft], ["train accuracy", "val accuracy"],"Accuracy autoencoder DeepFashion", axis2="Accuracy")

    # test with DeepFashion
    average_test_loss, average_test_accuracy= test_autoencoder_deep_fashion(encoder_ft, decoder_ft, classifier_ft)
    plot_summary([train_acc_ft, val_acc_ft], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy autoencoder FashionMNIST", axis2="Accuracy")
    plot_summary([train_losses_ft, val_losses_ft], average_test_loss, ["train loss", "val loss", "test average loss"], "Loss autoencoder FashionMNIST", axis2="Loss")
