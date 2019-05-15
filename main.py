"""Main"""
import time

from AET import transfer_learning_aet, test_aet, train_aet_mnist_cnn, train_aet_deep_fashion_cnn, \
    transfer_learning_aet_deep_fashion, test_aet_deep_fashion
from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn, \
    test_classification_on_exemplar_cnn, train_exemplar_cnn_deep_fashion, fine_tune_exemplar_cnn_deep_fashion, \
    test_classification_on_exemplar_cnn_deep_fashion
from rotation import train_rotation_net, fine_tune_rotation_model, \
    test_classification_on_rotation_model, train_rotation_net_deep_fashion, fine_tune_rotation_model_deep_fashion, \
    test_classification_on_rotation_model_deep_fashion
from supervised_deep_fashion import train_supervised_deep_fashion, train_supervised_FashionMNIST, \
    test_classification_deep_fashion, test_classification_on_supervised_fashionMNIST
from utils import plot_n_curves

since = time.time()

print("====================================")
print("========== Load the data ===========")
print("====================================\n")


################################################
#         AET sub task Fashion MNIST           #
################################################

# train AET with FashionMNIST
aet_f_trained, train_losses_aet_f = train_aet_mnist_cnn()
plot_n_curves(train_losses_aet_f, "Train losses", "Loss training AET Fashion MNIST", axis2="Loss")

# transfer learning with FashionMNIST
encoder, train_losses_aet, val_losses_aet, train_accuracies_aet, val_accuracies_aet = transfer_learning_aet(aet_f_trained)
plot_n_curves([train_losses_aet, val_losses_aet], ["train loss", "val loss"], "Loss AET FashionMNIST", axis2="Loss")
plot_n_curves([train_accuracies_aet, val_accuracies_aet], ["train accuracy", "val accuracy"], "Accuracy AET FashionMNIST", axis2="Accuracy")

# test with FashionMNIST
average_test_loss_ae_mnist, average_test_accuracy_ae_mnist = test_aet(encoder)
# plot_summary([train_accuracies_aet, val_accuracies_aet], average_test_accuracy_ae_mnist, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy test AET FashionMNIST", axis2="Accuracy") # TODO: check if it works
# plot_summary([train_losses_aet_f, val_losses_aet], average_test_loss_ae_mnist, ["train loss", "val loss", "test loss"], "Loss test AET FashionMNIST", axis2="Loss") # TODO: check if it works

################################################
#      AET sub task Fashion DeepFashion        #
################################################

# train AET with DeepFashion
aet_df_trained, train_losses_aet_df = train_aet_deep_fashion_cnn()
plot_n_curves(train_losses_aet_df, "Train losses", "Loss training AET DeepFashion", axis2="Loss")

# transfer learning with DeepFashion
encoder_df, train_losses_aet_df, val_losses_aet_df, train_accuracies_aet_df, val_accuracies_aet_df = transfer_learning_aet_deep_fashion(aet_df_trained)
plot_n_curves([train_losses_aet_df, val_losses_aet_df], ["train loss", "val loss"], "Loss AET FashionMNIST", axis2="Loss")
plot_n_curves([train_accuracies_aet_df, val_accuracies_aet_df], ["train accuracy", "val accuracy"], "Accuracy AET DeepFashion", axis2="Accuracy")

# test with DeepFashion
average_test_loss_ae_df, average_test_accuracy_ae_df = test_aet_deep_fashion(encoder_df)
# plot_summary([train_accuracies_aet_df, val_accuracies_aet_df], average_test_accuracy_ae_df, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test AET DeepFashion", axis2="Accuracy") # TODO: check if it works
# plot_summary([train_losses_aet_df, val_losses_aet_df], average_test_loss_ae_df, ["train loss", "val loss", "test loss"], "Loss Test AET DeepFashion", axis2="Loss") # TODO: check if it works

###############################################
#     Rotation sub task Fashion MNIST         #
###############################################

#train rotation net with FashionMNIST
rot_f_trained, train_losses_rot_f, val_losses_rot_f, train_acc_rot_f, val_acc_rot_f = train_rotation_net()
plot_n_curves([train_losses_rot_f, val_losses_rot_f], ["train loss", "val loss"], "Loss rotation FashionMNIST", axis2="Loss")
plot_n_curves([train_acc_rot_f, val_acc_rot_f], ["train accuracy", "val accuracy"], "Accuracy rotation FashionMNIST", axis2="Accuracy")

# fine tune rotation net with FashionMNIST
rot_f_finetuned, train_losses_rot_f_ft, val_losses_rot_f_ft, train_acc_rot_f_ft, val_acc_rot_f_ft = fine_tune_rotation_model(rot_f_trained)
plot_n_curves([train_losses_rot_f_ft, val_losses_rot_f_ft], ["train loss", "val loss"], "Loss fine tune rotation FashionMNIST", axis2="Loss")
plot_n_curves([train_acc_rot_f_ft, val_acc_rot_f_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune rotation FashionMNIST", axis2="Accuracy")

# test with FashionMNIST
average_test_loss_rot_mnist, average_test_accuracy_rot_mnist = test_classification_on_rotation_model(rot_f_finetuned)
# plot_n_curves(test_losses_rot_f, "Test loss", "Loss test rotation FashionMNIST", axis2="Loss") # TODO: fix
# plot_n_curves(test_acc_rot_f, "Test accuracy", "Accuracy test rotation FashionMNIST", axis2="Accuracy") # TODO: fix

###############################################
#     Rotation sub task DeepFashion           #
###############################################

# train rotation net with DeepFashion
rot_df_trained, train_losses_rot_df, val_losses_rot_df, train_acc_rot_df, val_acc_rot_df = train_rotation_net_deep_fashion()
plot_n_curves([train_losses_rot_df, val_losses_rot_df], ["train loss", "val loss"], "Loss rotation DeepFashion", axis2="Loss")
plot_n_curves([train_acc_rot_df, val_acc_rot_df], ["train accuracy", "val accuracy"], "Accuracy rotation DeepFashion", axis2="Accuracy")

# fine tune rotation net with DeepFashion
rot_df_finetuned, train_losses_rot_df_ft, val_losses_rot_df_ft, train_acc_rot_df_ft, val_acc_rot_df_ft = fine_tune_rotation_model_deep_fashion(rot_df_trained)
plot_n_curves([train_losses_rot_df_ft, val_losses_rot_df_ft], ["train loss", "val loss"], "Loss fine tune rotation DeepFashion", axis2="Loss")
plot_n_curves([train_acc_rot_df_ft, val_acc_rot_df_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune rotation DeepFashion", axis2="Accuracy")

# test with DeepFashion
average_test_loss_rot_df, average_test_accuracy_rot_df = test_classification_on_rotation_model_deep_fashion(rot_df_finetuned)
# plot_n_curves(test_losses_rot_df, "Test loss", "Loss test rotation DeepFashion", axis2="Loss") # TODO: fix
# plot_n_curves(test_acc_rot_df, "Test accuracy", "Accuracy test rotation DeepFashion", axis2="Accuracy") # TODO: fix

################################################
#     Exemplar CNN sub task Fashion MNIST      #
################################################

# train exemplar cnn with FashionMNIST
ex_cnn_f_trained, train_losses_ex_cnn_f, train_acc_ex_cnn_f = train_exemplar_cnn()
plot_n_curves([train_losses_ex_cnn_f], ["train loss"], "Loss train ExemplarCNN FashionMNIST", axis2="Loss")
plot_n_curves([train_acc_ex_cnn_f], ["train accuracy"], "Accuracy train ExemplarCNN FashionMNIST", axis2="Accuracy")

# fine tune exemplar cnn with FashionMNIST
ex_cnn_f_trained_finetuned, train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft, train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft = fine_tune_exemplar_cnn(ex_cnn_f_trained)
plot_n_curves([train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft], ["train loss", "val loss"], "Loss fine tune ExemplarCNN FashionMNIST", axis2="Loss")
plot_n_curves([train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune ExemplarCNN FashionMNIST", axis2="Accuracy")

# test with FashionMNIST
average_test_loss_ex_cnn_f, average_test_accuracy_ex_cnn_f = test_classification_on_exemplar_cnn(ex_cnn_f_trained_finetuned)
# plot_n_curves(average_test_accuracy_ex_cnn_f, "Test accuracy", "Accuracy test ExemplarCNN FashionMNIST", axis2="Accuracy") # TODO: fix
# plot_n_curves(average_test_loss_ex_cnn_f, "Test loss", "Loss test ExemplarCNN FashionMNIST", axis2="Loss") # TODO: fix

################################################
#     Exemplar CNN sub task Deep Fashion       #
################################################

# train exemplar cnn with DeepFashion
ex_cnn_df_trained, train_losses_ex_cnn_df, train_acc_ex_cnn_df = train_exemplar_cnn_deep_fashion()
plot_n_curves([train_losses_ex_cnn_df], ["train loss"], "Loss train ExemplarCNN DeepFashion", axis2="Loss")
plot_n_curves([train_acc_ex_cnn_df], ["train accuracy"], "Accuracy train ExemplarCNN DeepFashion", axis2="Accuracy")

# fine tune exemplar cnn with DeepFashion
ex_cnn_df_trained_finetuned, train_losses_ex_cnn_df_ft, val_loss_ex_cnn_df_ft, train_acc_ex_cnn_df_ft, val_acc_ex_cnn_df_ft = fine_tune_exemplar_cnn_deep_fashion(ex_cnn_df_trained)
plot_n_curves([train_losses_ex_cnn_df_ft, val_loss_ex_cnn_df_ft], ["train loss", "val loss"], "Loss fine tune ExemplarCNN DeepFashion", axis2="Loss")
plot_n_curves([train_acc_ex_cnn_df_ft, val_acc_ex_cnn_df_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune ExemplarCNN DeepFashion", axis2="Accuracy")

# test with DeepFashion
average_test_loss_ex_cnn_df, average_test_accuracy_ex_cnn_df = test_classification_on_exemplar_cnn_deep_fashion(ex_cnn_df_trained_finetuned)
# plot_n_curves(test_acc_ex_cnn_df, "Test accuracy", "Loss test ExemplarCNN DeepFashion", axis2="Accuracy") # TODO: fix
# plot_n_curves(test_losses_ex_cnn_df, "Test loss", "Loss test ExemplarCNN DeepFashion", axis2="Loss") # TODO: fix

################################################
#       Supervised subtask Fashion MNIST       #
################################################

# supervised training with Fashion MNIST
sv_mnist_trained, train_losses_sv_mnist, val_losses_sv_mnist, train_acc_sv_mnist, val_acc_sv_mnist = train_supervised_FashionMNIST()
plot_n_curves([train_losses_sv_mnist, val_losses_sv_mnist], ["train loss", "val loss"], "Loss train Supervised Fashion MNIST", axis2="Loss")
plot_n_curves([train_acc_sv_mnist, val_acc_sv_mnist], ["train accuracy", "val accuracy"], "Accuracy train supervised Fashion MNIST", axis2="Accuracy")

# test with Fashion MNIST
average_test_loss_sv_mnist, average_test_accuracy_sv_mnist = test_classification_on_supervised_fashionMNIST(sv_mnist_trained)
# plot_n_curves(test_losses_sv_mnist, "Test loss", "Loss test supervised Fashion MNIST", axis2="Loss") # TODO: fix
# plot_n_curves(test_acc_sv_mnist, "Test accuracy", "Accuracy test supervised Fashion MNIST", axis2="Accuracy") # TODO: fix

################################################
#       Supervised subtask Deep Fashion        #
################################################

# supervised training with DeepFashion
sv_df_trained, train_losses_sv_df, val_losses_sv_df, train_acc_sv_df, val_acc_sv_df = train_supervised_deep_fashion()
plot_n_curves([train_losses_sv_df, val_losses_sv_df], ["train loss", "val loss"], "Loss train supervised DeepFashion", axis2="Loss")
plot_n_curves([train_acc_sv_df, val_acc_sv_df], ["train accuracy", "val accuracy"], "Accuracy train supervised DeepFashion", axis2="Accuracy")

# # test with DeepFashion
average_test_loss_sv_df, average_test_accuracy_sv_df = test_classification_deep_fashion(sv_df_trained)
# plot_n_curves(test_losses_sv_df, "Test loss", "Loss test supervised DeepFashion", axis2="Loss") # TODO: fix
# plot_n_curves(test_acc_sv_df, "Test accuracy", "Accuracy test supervised DeepFashion", axis2="Accuracy") # TODO: fix


time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
