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

## Repository Structure

    model/
      cat_dog_model.keras     full trained model
      cat_dog.weights.h5      model weights
      training_curves.png     accuracy and loss curves
    src/
      check_data.py           counts images and checks for corrupt files
      train.py                trains and evaluates the model
      predict.py              predicts a single image
    requirements.txt

## How to Run

    pip install -r requirements.txt
    python src/train.py
    python src/predict.py path/to/image.jpg

Example output: `Prediction: dog (92.8% confidence)`

## Project Links

- Live demo (Vercel): TO BE ADDED
- Project report: TO BE ADDED
- Presentation: TO BE ADDED
- YouTube video: TO BE ADDED
- LinkedIn post: TO BE ADDED

## Author

Eshal Afzal Sial