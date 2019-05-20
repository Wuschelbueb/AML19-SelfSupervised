from autoencoder import train_autoencoder_deep_fashion, fine_tune_autoencoder_deep_fashion, \
    test_autoencoder_deep_fashion
from utils import plot_n_curves, plot_summary


################################################
#      AET sub task Fashion DeepFashion        #
################################################

def autoencoder_subtask_deep_fashion():
    # train AET with DeepFashion
    encoder_trained, losses = train_autoencoder_deep_fashion()
    plot_n_curves(losses, "Train losses", "Loss training autoencoder DeepFashion", axis2="Loss")

    # transfer learning with DeepFashion
    encoder_ft, train_losses, val_losses, train_acc, val_acc = fine_tune_autoencoder_deep_fashion(encoder_trained)

    # test with DeepFashion
    average_test_loss, average_test_accuracy = test_autoencoder_deep_fashion(encoder_ft)
    plot_summary([train_acc, val_acc], average_test_accuracy, ["train accuracy", "val accuracy", "test accuracy"],
                 "Accuracy autoencoder DeepFashion", axis2="Accuracy")
    plot_summary([train_losses, val_losses], average_test_loss, ["train loss", "val loss", "test average loss"],
                 "Loss autoencoder DeepFashion", axis2="Loss")

    return average_test_loss, average_test_accuracy
