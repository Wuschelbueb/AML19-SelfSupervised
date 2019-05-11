"""Main"""

from rotation import train_rotation_net, fine_tune_rotation_model
from exemplar_cnn import train_exemplar_cnn, fine_tune_exemplar_cnn

rotation_model_trained = train_rotation_net()
classification_model_1 = fine_tune_rotation_model(rotation_model_trained, False, False, False, True)

exemplar_cnn_trained = train_exemplar_cnn()
classification_model_2 = fine_tune_exemplar_cnn(exemplar_cnn_trained, False, False, False, True)
