import torch

from RotationModel import ResNet20, train_exemplar_cnn_model, train_rotation_model, fine_tune_fc, fine_tune_variant_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet20 = ResNet20()
resnet20 = resnet20.to(device)


print("Train ResNet20 with ExemplarCNN\n")
resnet20_trained = train_exemplar_cnn_model(resnet20)
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