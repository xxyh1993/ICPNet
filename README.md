## Title
A biological vision inspired framework for machine perception of illusory contours
![image](https://github.com/xxyh1993/ICPNet/blob/main/outline.png)

## Dataset
Downloading datasets from google drive and unzip it into your computer.

AG_MNIST: https://share.multcloud.link/share/4aec12b4-f2c1-4c8b-a85e-8e625933af4c

MNIST and its edge pseudo labels: https://share.multcloud.link/share/a41afedf-107f-44c0-9091-b56240b4ce51

Fashion_MNIST and AG_Fashion_MNIST: https://share.multcloud.link/share/985a693a-d31b-4df8-9eb4-71e90f82899b


## Environment
```bash
pytorch==2.0.1
numpy==1.22.4
python==3.9.17
torchvision==0.15.2 
pyyaml==6.0.1
einops==0.7.0
```

## model weights
The weights of all models can be downloaded from the link as follows:
https://share.multcloud.link/share/a1d22606-8572-46b7-b395-8c0c2b072139

Unzip the file into a directory. And then modifying the path of the models in ".yaml" files.


## Training and testing on MNIST and Fashion_MNIST
The training and test are integrated to a python file. You can also run the "all_models_test_illusory.py" to evaluate directly the performance of the pre-trained models on the AG_MNIST and AG_Fashion_MNIST test sets.

The “icpnet_train_classifier_mnist.py” is the training and test of ICPNet on MNIST and AG_MNIST.

The "icpnet_train_classifier_fashion_mnist.py" is the training and test of ICPNet on Fashion_MNIST and AG_Fashion_MNIST.

The "other_model_training.py" is the training and test of other models on Fashion_MNIST, AG_Fashion_MNIST, MNIST, and AG_MNIST.

The "icpnet_model.py" is the constructed model.

The "visual_loss_edge2.py" is the visualization file based on visdom

