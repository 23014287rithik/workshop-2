# workshop2
# Building an AI Classifier: Identifying Cats, Dogs & Pandas with PyTorch

This repository contains a deep learning project that builds an image classifier to differentiate between images of cats, dogs, and pandas. The model is implemented using **PyTorch** and leverages **Transfer Learning** with a pre-trained ResNet18 model to achieve high accuracy with minimal training time.

## 📁 Project Structure

*   **`DL_Workshop2.ipynb`**: The main Jupyter Notebook containing the complete workflow, including data downloading, preprocessing, model definition, training loop, and evaluation.
*   **`requirements.txt`**: A list of all necessary Python libraries to run the project.

## 🚀 Features

*   **Automated Data Pipeline**: Downloads the [Animal Detection Dataset (Cats, Dogs, and Pandas)](https://www.kaggle.com/datasets/sohampatel26/animal-detection-dataset-cats-dogs-and-pandas) directly from Kaggle.
*   **Data Augmentation**: Uses `torchvision.transforms` for random cropping, horizontal flipping, and rotation to make the model more robust.
*   **Transfer Learning**: Fine-tunes a `ResNet18` model (pre-trained on ImageNet) by freezing the convolutional base and replacing the classifier head.
*   **Training and Evaluation**: Implements a custom PyTorch training loop with loss tracking, accuracy calculations, and saving the best performing model (`best_model.pth`).

## 🛠️ Requirements

Make sure you have Python installed, then install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

**Note**: To download the dataset automatically within the notebook, you will need a valid `kaggle.json` API token configured on your machine or Colab environment.

## 🧠 Model Architecture

*   **Base Model**: ResNet18 (frozen weights)
*   **Custom Head**:
    *   Linear layer (`in_features` -> 256)
    *   ReLU Activation
    *   Dropout (p=0.5)
    *   Linear layer (256 -> 3 classes)
*   **Loss Function**: Cross-Entropy Loss
*   **Optimizer**: Adam (learning rate = 0.001)

## 💻 Usage

1.  Clone this repository.
2.  Install the required dependencies.
3.  Open `DL_Workshop2.ipynb` using Jupyter Notebook or Google Colab.
4.  Run the cells sequentially to train the model and see the evaluation metrics.
