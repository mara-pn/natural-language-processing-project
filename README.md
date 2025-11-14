# CIFAR-10 Classification Project (Image Recognition)

This project explores image recognition using the CIFAR-10 dataset, a benchmark dataset for object classification.
A range of Convolutional Neural Network (CNN) architectures were implemented — from baseline models to transfer learning approaches with pre-trained ImageNet networks such as MobileNetV2, EfficientNetB0, and DenseNet121.

## Project Overview

This notebook documents the process of:
- Loading and visualizing the CIFAR-10 dataset
- Building and evaluating baseline CNN architectures
- Incrementally improving model performance through architectural modifications (e.g., Batch Normalization, Global Average Pooling, Dropout)
- Experimenting with different optimizers (Adam, SGD)
- Implementing transfer learning with pre-trained models from ImageNet
- Fine-tuning and data augmentation for improved generalization
- Comparing performance metrics across all tested models

The final results are summarized in a comparison table, showing model accuracies and loss values across configurations.

## Dataset Information

Dataset: CIFAR-10

Creators:
- Alex Krizhevsky
- Vinod Nair
- Geoffrey Hinton

Institution: University of Toronto

## Description:
The CIFAR-10 dataset consists of 60,000 32×32 color images divided into 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

Each class contains 6,000 images. The classes are mutually exclusive (e.g., “automobile” and “truck” do not overlap).

## Methods & Models

Convolutional Neural Networks:
- Baseline CNN: Simple two-layer architecture for reference performance
- Enhanced CNNs: Deeper networks with Batch Normalization, Dropout, and GAP layers
- Optimizers: Comparative runs with Adam and SGD

Transfer Learning:
- MobileNetV2 (frozen & fine-tuned)
- DenseNet121 (frozen, fine-tuned, and fine-tuned with augmentation)
- EfficientNetB0 (exploratory, referenced)

All models were trained and evaluated using TensorFlow/Keras with standard preprocessing and one-hot encoding.

## Evaluation & Visualization

The notebook includes:
- Training and validation accuracy/loss plots
- Confusion matrices for visualizing misclassifications
- Example visualizations of “confusing class pairs” (e.g., cat ↔ dog)
- Comparative validation accuracy/loss charts across models

Final results are compiled in a sorted performance table.

## Collaboration & Contributions

This project was developed in collaboration with several peers, where different team members focused on different aspects of model selection and evaluation.

Model selection work was distributed among the team, meaning that the best-performing results were obtained in a different notebook as part of this shared project effort (see project presentation (CIFAR-10_Classification_Project.pdf)).

## References

Krizhevsky, A., & Hinton, G. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto.

TensorFlow Documentation: https://www.tensorflow.org/api_docs

Keras Applications: https://keras.io/api/applications/

CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## License

This work uses the CIFAR-10 dataset, © University of Toronto, used under fair academic and research use.
All model implementations and visualizations are released under the MIT License unless otherwise stated.