# action_recognition
# A simple implementation of action recognition using Gluon CV Toolkit

Since GluonCV already provides implementations of the state-of-the-art deep learning models in computer vision in areas of image classification, object detection, segmentation, pose estimation, action recognition, object tracking and depth prediction. I decided to leverage on GluonCV for the following:

1. Training scripts to reproduce SOTA results reported in research papers
2. A large number of pre-trained models
3. APIs that greatly reduce the implementation complexity

This Repository contains a simple implementation of action recognition using a pre-trained model "i3d_resnet50_v1_kinetics400" trained on the kinetics400 dataset.
Main reason to choose this model is down to a good trade off in inferencing accuracy, inferencing speed and model size after experimenting a few of the SOTA models.


# Installation

GluonCV is built on top of MXNet and PyTorch. For my use case i used MXNet

## Installation (MXNet)

GluonCV supports Python 3.6 or later.

```bash
pip install gluoncv --upgrade
# native
pip install -U --pre mxnet -f https://dist.mxnet.io/python/mkl
# cuda 10.2
pip install -U --pre mxnet -f https://dist.mxnet.io/python/cu102mkl
pip install decord
```


