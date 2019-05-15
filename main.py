"""Main"""

from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn, \
    test_classification_on_exemplar_cnn
from rotation import train_rotation_net, fine_tune_rotation_model, \
    test_classification_on_rotation_model
from supervised_deep_fashion import train_supervised_deep_fashion, train_supervised_FashionMNIST

print("====================================")
print("========== Load the data ===========")
print("====================================\n")

################################################
#               Rotation sub task              #
################################################

# train rotation net with FashionMNIST
rot_f_trained, train_losses_rot_f, val_losses_rot_f, train_acc_rot_f, val_acc_rot_f = train_rotation_net()
# plot_n_curves([train_losses_rot_f, val_losses_rot_f], ["rain loss", "val loss"], "Loss rotation FashionMNIST")
# plot_n_curves([train_acc_rot_f, val_acc_rot_f], ["train accuracy", "val accuracy"], "Accuracy rotation FashionMNIST")

# train rotation net with DeepFashion
# rot_df_trained, train_losses_rot_df, val_losses_rot_df, train_acc_rot_df, val_acc_rot_df = train_rotation_net_deep_fashion()
# plot_n_curves([train_losses_rot_df, val_losses_rot_df], ["train loss", "val loss"], "Loss rotation DeepFashion")
# plot_n_curves([train_acc_rot_df, val_acc_rot_df], ["train accuracy", "val accuracy"], "Accuracy rotation DeepFashion")

# fine tune rotation net with FashionMNIST
rot_f_finetuned, train_losses_rot_f_ft, val_losses_rot_f_ft, train_acc_rot_f_ft, val_acc_rot_f_ft = fine_tune_rotation_model(rot_f_trained)
# plot_n_curves([train_losses_rot_f_ft, val_losses_rot_f_ft], ["train loss", "val loss"], "Loss fine tune rotation FashionMNIST")
# plot_n_curves([train_acc_rot_f_ft, val_acc_rot_f_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune rotation FashionMNIST")

# fine tune rotation net with DeepFashion
# rot_df_finetuned, train_losses_rot_df_ft, val_losses_rot_df_ft, train_acc_rot_df_ft, val_acc_rot_df_ft = fine_tune_rotation_model_deep_fashion(rot_df_trained)
# plot_n_curves([train_losses_rot_df_ft, val_losses_rot_df_ft], ["train loss", "val loss"], "Loss fine tune rotation DeepFashion")
# plot_n_curves([train_acc_rot_df_ft, val_acc_rot_df_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune rotation DeepFashion")

# test with FashionMNIST
test_losses_rot_f, test_acc_rot_f = test_classification_on_rotation_model(rot_f_finetuned)
# plot_n_curves(test_losses_rot_f, "Test loss", "Loss test rotation FashionMNIST")
# plot_n_curves(test_acc_rot_f, "Test accuracy", "Accuracy test rotation FashionMNIST")

# test with DeepFashion
# test_losses_rot_df, test_acc_rot_df = test_classification_on_rotation_model_deep_fashion(rot_df_finetuned)
# plot_n_curves(test_losses_rot_df, "Test loss", "Loss test rotation DeepFashion")
# plot_n_curves(test_acc_rot_df, "Test accuracy", "Accuracy test rotation DeepFashion")

################################################
#           Exemplar CNN sub task              #
################################################

# train exemplar cnn with FashionMNIST
ex_cnn_f_trained, train_losses_ex_cnn_f, train_acc_ex_cnn_f = train_exemplar_cnn()
# plot_n_curves([train_losses_ex_cnn_f], ["train loss"], "Loss train ExemplarCNN FashionMNIST")
# plot_n_curves([train_acc_ex_cnn_f], ["train accuracy"], "Accuracy train ExemplarCNN FashionMNIST")

# train exemplar cnn with DeepFashion
# ex_cnn_df_trained, train_losses_ex_cnn_df, train_acc_ex_cnn_df = train_exemplar_cnn_deep_fashion()
# plot_n_curves([train_losses_ex_cnn_df], ["train loss"], "Loss train ExemplarCNN DeepFashion")
# plot_n_curves([train_acc_ex_cnn_df], ["train accuracy"], "Accuracy train ExemplarCNN DeepFashion")

# fine tune exemplar cnn with FashionMNIST
ex_cnn_f_trained_finetuned, train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft, train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft = fine_tune_exemplar_cnn(ex_cnn_f_trained)
# plot_n_curves([train_losses_ex_cnn_f_ft, val_loss_ex_cnn_f_ft], ["train loss", "val loss"], "Loss fine tune ExemplarCNN FashionMNIST")
# plot_n_curves([train_acc_ex_cnn_f_ft, val_acc_ex_cnn_f_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune ExemplarCNN FashionMNIST")

# fine tune exemplar cnn with DeepFashion
# ex_cnn_df_trained_finetuned, train_losses_ex_cnn_df_ft, val_loss_ex_cnn_df_ft, train_acc_ex_cnn_df_ft, val_acc_ex_cnn_df_ft = fine_tune_exemplar_cnn_deep_fashion(ex_cnn_df_trained)
# plot_n_curves([train_losses_ex_cnn_df_ft, val_loss_ex_cnn_df_ft], ["train loss", "val loss"], "Loss fine tune ExemplarCNN DeepFashion")
# plot_n_curves([train_acc_ex_cnn_df_ft, val_acc_ex_cnn_df_ft], ["train accuracy", "val accuracy"], "Accuracy fine tune ExemplarCNN DeepFashion")

# test with FashionMNIST
test_losses_ex_cnn_f, test_acc_ex_cnn_f = test_classification_on_exemplar_cnn(ex_cnn_f_trained_finetuned)
# plot_n_curves(test_acc_ex_cnn_f, "Test accuracy", "Accuracy test ExemplarCNN FashionMNIST")
# plot_n_curves(test_losses_ex_cnn_f, "Test loss", "Loss test ExemplarCNN FashionMNIST")

# test with DeepFashion
# test_losses_ex_cnn_df, test_acc_ex_cnn_df = test_classification_on_exemplar_cnn_deep_fashion(ex_cnn_df_trained_finetuned)
# plot_n_curves(test_acc_ex_cnn_df, "Test accuracy", "Loss test ExemplarCNN DeepFashion")
# plot_n_curves(test_losses_ex_cnn_df, "Test loss", "Loss test ExemplarCNN DeepFashion")

################################################
#           Supervised subtask                 #
################################################

# supervised training with DeepFashion
sv_cnn_df_trained, train_losses_sv_cnn_df, train_acc_sv_cnn_df = train_supervised_deep_fashion()
# plot_n_curves([train_losses_sv_cnn_df], ["train loss"], "Loss train supervised DeepFashion")
# plot_n_curves([train_acc_sv_cnn_df], ["train accuracy"], "Accuracy train supervised DeepFashion")

# # test with DeepFashion
# test_losses_sv_df, test_acc_sv_df = test_classification_deep_fashion(sv_cnn_df_trained)
# plot_n_curves(test_losses_sv_df, "Test loss", "Loss test supervised DeepFashion")
# plot_n_curves(test_acc_sv_df, "Test accuracy", "Accuracy test supervised DeepFashion")


# supervised training with Fashion MNIST
sv_mnist_trained, train_losses_sv_mnist, val_losses_sv_mnist, train_acc_sv_mnist, val_acc_sv_mnist = train_supervised_FashionMNIST()
#plot_n_curves([train_losses_sv_mnist, train_acc_sv_mnist], ["train loss"], "Loss train Supervised Fashion MNIST")
# plot_n_curves([train_acc_sv_mnist], ["train accuracy"], "Accuracy train supervised Fashion MNIST")

# test with Fashion MNIST
# test_losses_sv_mnist, test_acc_sv_mnist = test_classification_on_supervised_fashionMNIST(sv_mnist_trained)
# plot_n_curves(test_losses_sv_mnist, "Test loss", "Loss test supervised Fashion MNIST")
# plot_n_curves(test_acc_sv_mnist, "Test accuracy", "Accuracy test supervised Fashion MNIST")