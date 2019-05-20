"""Main"""
import time

from autoencoder_subtask_Deep_Fashion import autoencoder_subtask_deep_fashion
from autoencoder_subtask_Fashion_MNIST import autoencoder_subtask_fashion_mnist
from exemplar_cnn_subtask_Deep_Fashion import exemplar_cnn_subtask_deep_fashion
from exemplar_cnn_subtask_Fashion_MNIST import exemplar_cnn_subtask_fashion_mnist
from rotation_subtask_Deep_Fashion import rotation_subtask_deep_fashion
from rotation_subtask_Fashion_MNIST import rotation_subtask_fashion_mnist
from supervised_Deep_Fashion import supervised_deep_fashion
from supervised_Fashion_MNIST import supervised_fashion_mnist

since = time.time()

print("====================================")
print("========== Load the data ===========")
print("====================================\n")

ae_fm_average_test_loss, ae_fm_average_test_accuracy = autoencoder_subtask_fashion_mnist()
ae_df_average_test_loss, ae_df_average_test_accuracy = autoencoder_subtask_deep_fashion()
rot_fm_average_test_loss, rot_fm_average_test_accuracy = rotation_subtask_fashion_mnist()
rot_df_average_test_loss, rot_df_average_test_accuracy = rotation_subtask_deep_fashion()
ex_fm_average_test_loss, ex_fm_average_test_accuracy = exemplar_cnn_subtask_fashion_mnist()
ex_df_average_test_loss, ex_df_average_test_accuracy = exemplar_cnn_subtask_deep_fashion()
sv_fm_average_test_loss, sv_fm_average_test_accuracy = supervised_fashion_mnist()
sv_df_average_test_loss, sv_df_average_test_accuracy = supervised_deep_fashion()

print('Test average loss AE - Fashion MNIST:', ae_fm_average_test_loss, 'Test accuracy AE - Fashion MNIST:', ae_fm_average_test_accuracy)
print('Test average loss AE - Deep Fashion:', ae_df_average_test_loss, 'Test accuracy AE - Deep Fashion:', ae_df_average_test_accuracy)
print('Test average loss rotation - Fashion MNIST:', rot_fm_average_test_loss, 'Test accuracy rotation - Fashion MNIST:', rot_fm_average_test_accuracy)
print('Test average loss rotation - Deep Fashion:', rot_df_average_test_loss, 'Test accuracy rotation - Deep Fashion:', rot_df_average_test_accuracy)
print('Test average loss exemplarCNN - Fashion MNIST:', ex_fm_average_test_loss, 'Test accuracy exemplarCNN - Fashion MNIST:', ex_fm_average_test_accuracy)
print('Test average loss exemplarCNN - Deep Fashion:', ex_df_average_test_loss, 'Test accuracy exemplarCNN - Deep Fashion:', ex_df_average_test_accuracy)
print('Test average loss supervised - Fashion MNIST:', sv_fm_average_test_loss, 'Test accuracy supervised - Fashion MNIST:', sv_fm_average_test_accuracy)
print('Test average loss supervised - Deep Fashion:', sv_df_average_test_loss, 'Test accuracy supervised - Deep Fashion:', sv_df_average_test_accuracy)

time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
