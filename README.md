# AML19-SelfSupervised

Self Supervised Learning: Fashion Classification

Team members

    Ramona Beck

    David Berger

    Gwenael Gendre

    Nathalie Froidevaux

Project description

    Datasets: 
        - FashionMNIST: https://github.com/zalandoresearch/fashion-mnist
        - DeepFashion: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

    Task: Classification of different fashion items with Self Supervised Learning

    Architecture
        - CifarNet
        
    Pretrainig
        - Image Rotation Predictions
        - Exemplar Convolutional Neural Networks (Exemplar CNN)
        - Autoencoder
        
 
 
        

## Data
The FashionMNIST data set provides 60'000 train and 10'000 test grayscale images with size 28x28, each one is associated with one of the 10 labels.

The DeepFashion data set provides almost 300'000 images. The images are colorized and of different size, and each image is associated with one of 50 possible labels amongst others.


## Pretraining
We use different pre-training approaches to compare which approach leads to the best classification accuracy on the data.

### Rotation Pre-Training
For this task, we rotate every picture in the batch 4 times (0, 90, 180 and 270 degrees), associate the image with the angle and try to predict the rotation.

### Exemplar CNN Pre-Training
In this task, we apply 6 different transformations to every image in the batch. The original image is associated with it's own class and all different versions of the original image are associated with the same class, so in the end in each batch we have as many different classes as the batch size and to every class belong 6 differently transformed images. 

### Autoencoder
In this task we use the CifarNet as encoder. We don't do any data augmentation for this task.


## Fine Tuning
After the pre-training we finetune our models. Therefore we "unfreeze" the final fully connected layer of the CifarNet.


## Classification
In the end we do the fashion classification task on our fine-tuned models.
