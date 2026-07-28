# An Interactive Tutorial for Multi-Instance Contrastive Learning (MICLe)

A from-scratch implementation of the Multi-Instance Contrastive Learning (MICLe) model from Azizi et al., ["Big Self-Supervised Models Advance Medical Image Classification" (2021)](https://arxiv.org/abs/2101.05224), built in TensorFlow/Keras.

MICLe is a self-supervised pretraining method for settings where a dataset contains multiple images of the same underlying case, for example, several views of the same pathology in medical imaging. It forms positive pairs from images of the same case and learns representations that are resilient to changes in view, lighting, and other variation, without requiring labels.

## What's inside

`MICLe.ipynb` is an interactive tutorial that:
- explains the motivation and the MICLe algorithm, including contrastive loss.
- implements the full pipeline in TensorFlow/Keras: bag-based data arrangement, a random-crop augmenter, an encoder, and a non-linear projection head.
- builds MICLe as a custom `keras.Model` with a contrastive `train_step`, and trains it on a multi-view demonstration dataset (objects photographed from multiple angles, standing in for the multi-view medical imaging case).

## Getting started

Install dependencies:
```
pip install -r requirements.txt
```

## Usage 

Follow the tutorial using the notebook `MICLe.ipynb`:
```
jupyter notebook MICLe.ipynb
```

`MICLe.ipynb` requires `encodeutil.py` (included), which defines the base encoder network. The demonstration dataset is included under `data/unlabeled_images/`, with one subfolder per "bag" (a set of images of the same object).

## Dependencies
- NumPy
- Matplotlib
- TensorFlow