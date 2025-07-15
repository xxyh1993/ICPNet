## Title
A biological vision inspired framework for machine perception of abutting grating illusory contours


## Dataset
Downloading datasets from google drive and unzip it to your computer.

AG_MNIST: https://drive.google.com/file/d/1831nP9CLYym_cX53Tta7ndgGN2yyYMl4/view?usp=drive_link

MNIST and its edge pseudo labels: https://drive.google.com/file/d/19WcR-1PwrztavkrzvlrtxhQNL-Z8eUZP/view?usp=drive_link 

Fashion_MNIST and AG_Fashion_MNIST: https://drive.google.com/file/d/11tp84BY0vA_8mbrihTGMu6nW6ZvgLODP/view?usp=drive_link


## Environment
```
pytorch==2.2.1
numpy==1.24.4
opencv-python==4.2.0
scikit-image==0.21.0
```

## Training and testing on MNIST and Fashion_MNIST
The training and test are integrated to a python file. You can also run the "all_models_test_illusory.py" to evaluate directly the performance of the pre-trained models on the AG_MNIST and AG_Fashion_MNIST test sets.

The “icpnet_train_classifier_mnist.py” is the training and test of ICPNet on MNIST and AG_MNIST.

The "icpnet_train_classifier_fashion_mnist.py" is the training and test of ICPNet on Fashion_MNIST and AG_Fashion_MNIST.

The "other_model_training.py" is the training and test of other models on Fashion_MNIST, AG_Fashion_MNIST, MNIST, and AG_MNIST.

The "icpnet_model.py" is the constructed model.

The "visual_loss_edge2.py" is the visualization file based on visdom

