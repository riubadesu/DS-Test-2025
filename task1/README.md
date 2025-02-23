# MNIST Classification Project

This project implements three different machine learning models for handwritten digit recognition using the MNIST dataset.

## Project Overview

The following classifiers are trained and tested for accuracy and robustness:

- **Random Forest (RF)** – A decision-tree-based machine learning model.
- **Feed-Forward Neural Network (FFNN)** – A simple deep learning model using dense layers.
- **Convolutional Neural Network (CNN)** – A deep learning model optimized for image classification.

Each model is tested for accuracy, robustness, and edge case handling.

## Project Structure

```
task1/
│── mnist_classification.ipynb    # Main Jupyter Notebook (combines all models)
│── mnist_rf.ipynb                # Random Forest Implementation
│── mnist_ffnn.ipynb              # Feed-Forward NN Implementation
│── mnist_cnn.ipynb               # Convolutional NN Implementation
│── requirements.txt               # Dependencies list for easy installation
│── README.md                      # Project documentation (this file)
```

## Installation

### 1. Clone the Repository (If Using Git)
```sh
git clone https://github.com/riubadesu/DS-Test-2025.git
cd mnist-classification/task1
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook
```sh
jupyter lab
```
Then open **`mnist_classification.ipynb`** to run the code step by step.

This project is **cross-platform** and runs on:
- Windows (via Command Prompt, PowerShell, WSL, or Git Bash)
- Linux (Ubuntu, Debian, Fedora, etc.)
- macOS (Intel and M1/M2 chips)

## Usage

The `MnistClassifier` wrapper allows switching between different models.

### Example: Train and Predict Using CNN
```python
from mnist_classification import MnistClassifier

# Choose a model ('rf' for Random Forest, 'nn' for FFNN, 'cnn' for CNN)
clf = MnistClassifier(algorithm='cnn')

# Train the model
clf.train(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
print(predictions[:10])  # Show first 10 predictions
```

## Model Performance

Each model is trained and tested in **`mnist_classification.ipynb`**.  
Here are the expected accuracy results for the MNIST dataset:

| Model  | Expected Accuracy |
|--------|------------------|
| Random Forest (RF) | ~92% |
| Feed-Forward Neural Network (FFNN) | ~97% |
| Convolutional Neural Network (CNN) | ~98% |

## Edge Case Handling

To ensure the models handle unexpected inputs properly, the following cases were tested:

| **Edge Case** | **Expected Behavior** | **Status** |
|--------------|------------------|---------|
| Invalid Data Type (e.g., string input) | Should raise an error | Passed |
| Empty Data | Should raise an error | Passed |
| Wrong Input Shape (CNN requires 28×28 images) | Should raise an error | Passed |
| NaN Values in Training Data | Should raise an error | Passed |
| Invalid Labels (non-numeric) | Should raise an error | Passed |
| Batch Size of 1 (Minimal Input) | Model should train | Passed |

### Example: Handling Invalid Inputs
```python
try:
    clf = MnistClassifier(algorithm='rf')
    clf.train("invalid_data", "invalid_labels")  # Should raise an error
except Exception as e:
    print(f"Error Caught: {e}")
```

