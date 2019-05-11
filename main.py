"""Main"""

from rotation import train_rotation_net, fine_tune_rotation_model
from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn
from utils import plot_n_curves


rotation_trained, train_losses_rot, val_losses_rot, train_accuracies_rot, val_accuracies_rot = train_rotation_net()
plot_n_curves([train_losses_rot, val_losses_rot], ["train loss", "val loss"], "Loss for rotation task")
plot_n_curves([train_accuracies_rot, val_accuracies_rot], ["train accuracy", "val accuracy"],
              "Accuracy for rotation task")

# classification_model_1, train_losses, val_losses, train_accuracies, val_accuracies = fine_tune_rotation_model(
#     rotation_trained, False, False, False, True)


exemplar_cnn_trained, train_losses_ex, train_accuracies_ex = train_exemplar_cnn()
plot_n_curves([train_losses_ex], ["train loss"], "Loss for ExemplarCNN")
plot_n_curves([train_accuracies_ex], ["train accuracy"], "Accuracy for ExemplarCNN")

# classification_model_2, train_losses_c, val_losses, train_accuracies, val_accuracies = fine_tune_exemplar_cnn(
#     exemplar_cnn_trained, False, False, False, True)
