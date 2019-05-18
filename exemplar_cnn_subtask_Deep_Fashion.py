from exemplar_cnn import test_classification_on_exemplar_cnn_deep_fashion, fine_tune_exemplar_cnn_deep_fashion, \
    train_exemplar_cnn_deep_fashion
from utils import plot_summary, plot_n_curves

################################################
#     Exemplar CNN sub task Deep Fashion       #
################################################


def exemplar_cnn_subtask_deep_fashion():
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
    plot_summary([train_acc_ex_cnn_df_ft, val_acc_ex_cnn_df_ft], average_test_accuracy_ex_cnn_df, ["train accuracy", "val accuracy", "test accuracy"], "Accuracy Test ExemplarCNN Deep Fashion", axis2="Accuracy")
    plot_summary([train_losses_ex_cnn_df_ft, val_loss_ex_cnn_df_ft], average_test_loss_ex_cnn_df, ["train loss", "val loss", "test loss"], "Loss Test ExemplarCNN Deep Fashion", axis2="Loss")

    return average_test_loss_ex_cnn_df, average_test_accuracy_ex_cnn_df
