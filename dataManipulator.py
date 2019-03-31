
import torch
from torchvision import transforms


def preprocess(listOfNPArrayImages):

    transformation = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                               ])

    tensorList = []
    for img in listOfNPArrayImages:
        tensorList.append(transformation(img))

    stackedTensor = torch.stack(tensorList)

    return stackedTensor

