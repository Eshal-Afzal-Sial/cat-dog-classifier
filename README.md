# Cat vs Dog Image Classifier

A deep learning project that classifies images as **cat** or **dog** using transfer learning with MobileNetV2 (TensorFlow/Keras).

## Results

| Metric | Value |
|---|---|
| Test accuracy | 98.86% |
| Best validation accuracy | 99.38% |
| Test loss | 0.0344 |

## Dataset

- **Name:** Cat and Dog (by tongpython, Kaggle)
- **Link:** https://www.kaggle.com/datasets/tongpython/cat-and-dog
- **Size:** about 10,000 images in two balanced classes
  - Train: 4,000 cats, 4,005 dogs (80% used for training, 20% for validation)
  - Test: 1,011 cats, 1,012 dogs
- The images are not stored in this repository. Download them from the link above and extract to `dataset/data/` so the paths are `dataset/data/training_set/training_set` and `dataset/data/test_set/test_set`.

## Model

- Base: MobileNetV2 pre-trained on ImageNet (frozen)
- Head: Global Average Pooling, Dropout (0.2), Dense (1, sigmoid)
- Augmentation: random horizontal flip, rotation, zoom
- Optimizer: Adam (lr 1e-3), loss: binary cross-entropy
- Input size: 224 x 224
- Class labels: 0 = cat, 1 = dog

##