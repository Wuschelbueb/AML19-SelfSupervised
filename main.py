import torch
from CifarNet import CifarNet


from RotationModel import ResNet20ExemplarCNN, train_exemplar_cnn_model, train_rotation_model, fine_tune_fc, \
    fine_tune_variant_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cifarModel = CifarNet(inputChannels=1, numClasses=4)


pretrain_cifarModel = train_rotation_model(cifarModel)

# pretrain_resnet20 = train_rotation_model(resnet20)
# classification_resnet20 = fine_tune_fc(pretrain_resnet20)


print("Train ResNet20 with ExemplarCNN\n")
#resnet_20_exemplar_cnn_trained = train_exemplar_cnn_model(resnet_20_exemplar_cnn)
print("\n=========================================\n")


# print("Train ResNet20\n")
# resnet20_trained = train_rotation_model(resnet20)
# print("\n=========================================\n")
#
# print("\nFinetune ResNet20 (only linear layer)\n")
# resnet20_finetuned_1 = fine_tune_fc(resnet20_trained)
# print("\n=========================================\n")
#
# print("\nFinetune ResNet20 (layer 3 + linear layer)\n")
# resnet20_finetuned_2 = fine_tune_variant_2(resnet20_trained)
# print("\n=========================================\n")