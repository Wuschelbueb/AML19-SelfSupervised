"""Main"""
import time

from autoencoder_subtask_Deep_Fashion import autoencoder_subtask_deep_fashion
from autoencoder_subtask_Fashion_MNIST import autoencoder_subtask_fashion_MNIST
from exemplar_cnn_subtask_Deep_Fashion import exemplar_cnn_subtask_deep_fashion
from exemplar_cnn_subtask_Fashion_MNIST import exemplar_cnn_subtask_fashion_mnist
from rotation_subtask_Deep_Fashion import rotation_subtask_deep_fashion
from rotation_subtask_Fashion_MNIST import rotation_subtask_fashion_mnist

since = time.time()

print("====================================")
print("========== Load the data ===========")
print("====================================\n")

autoencoder_subtask_fashion_MNIST()
autoencoder_subtask_deep_fashion()
rotation_subtask_fashion_mnist()
rotation_subtask_deep_fashion()
exemplar_cnn_subtask_fashion_mnist()
exemplar_cnn_subtask_deep_fashion()
# supervised_fashion_mnist()
# supervised_deep_fashion()

time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
