"""Main"""

from rotation import train_rotation_net, fine_tune_rotation_model, test_classification_on_rotation_model
from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn, test_classification_on_exemplar_cnn
from utils import plot_n_curves

################################################
#               Rotation sub task              #
################################################

# train rotation net
rotation_trained, train_losses_rot, val_losses_rot, train_accuracies_rot, val_accuracies_rot = train_rotation_net()
plot_n_curves([train_losses_rot, val_losses_rot], ["train loss", "val loss"], "Loss for rotation task")
plot_n_curves([train_accuracies_rot, val_accuracies_rot], ["train accuracy", "val accuracy"],
              "Accuracy for rotation task")

# fine tune rotation net
rotation_finetuned, train_loss_ft_rot, val_loss_ft_rot, train_acc_ft_rot, val_acc_ft_rot = fine_tune_rotation_model(
    rotation_trained, False, False, True)
plot_n_curves([train_loss_ft_rot, val_loss_ft_rot], ["train loss", "val loss"], "Loss for tuning rotation model")
plot_n_curves([train_acc_ft_rot, val_acc_ft_rot], ["train accuracy", "val accuracy"],
              "Accuracy for fine tuning rotation model")

# test
test_loss_rot, test_accuracy_rot = test_classification_on_rotation_model(rotation_finetuned)
plot_n_curves(test_loss_rot, "Test accuracy", "Test accuracy of classification task on rotation net")
plot_n_curves(test_accuracy_rot, "Test loss", "Test loss of classification task on rotation net")

################################################
#           Exemplar CNN sub task              #
################################################

# train exemplar cnn
exemplar_cnn_trained, train_losses_ex, train_accuracies_ex = train_exemplar_cnn()
plot_n_curves([train_losses_ex], ["train loss"], "Loss for ExemplarCNN")
plot_n_curves([train_accuracies_ex], ["train accuracy"], "Accuracy for ExemplarCNN")

# fine tune exemplar cnn
exemplarcnn_finetuned, train_loss_ft_ex, val_loss_ft_ex, train_acc_ft_ex, val_acc_ft_ex = fine_tune_exemplar_cnn(
    exemplar_cnn_trained, False, False, True)
plot_n_curves([train_loss_ft_ex, val_loss_ft_ex], ["train loss", "val loss"], "Loss for fine tuning exemplar cnn")
plot_n_curves([train_acc_ft_ex, val_acc_ft_ex], ["train accuracy", "val accuracy"],
              "Accuracy for fine tuning exemplar cnn")

# test
test_loss_ex, test_accuracy_ex = test_classification_on_exemplar_cnn(exemplarcnn_finetuned)
plot_n_curves(test_loss_ex, "Test accuracy", "Test accuracy of classification task on exemplar cnn")
plot_n_curves(test_accuracy_ex, "Test loss", "Test loss of classification task on exemplar cnn")
