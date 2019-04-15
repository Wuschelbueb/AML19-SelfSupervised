# AML19-SelfSupervised

Self Supervised Learning: Fashionset Classification

Team members

    Ramona Beck

    David Berger

    Gwenael Gendre

    Nathalie Froidevaux

Project description

    Dataset: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

    Task: Classification of different fashion items with Self Supervised Learning

    Architecture
        ResNet-101 (https://pytorch.org/docs/stable/torchvision/models.html)
        
    Pretrainig
        - Image Rotation Predictions
        - Maybe Colorization or other pretraining methods, still TBD
        

We'll first test our implementation on the MNIST (https://github.com/zalandoresearch/fashion-mnist) to get an idea about the performance.


dev-david branch:
-to execute the programm you need to start the main.py file with the following parameters: "--exp=CIFAR10_RotNet_NIN4blocks", "--cuda=True or False"
-the fashionmnist dataset needs to be extracted into the data folder.
-on windows i can't get the programm up and running, as soon as i hit the first epoch, i get a pickle error. the source of the error seems to be in the algorithm.py file on line 287. maybe one of you knows a solution?