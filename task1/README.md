# MNIST Digit Classification Project

## Overview
This project implements three different models to classify handwritten digits from the MNIST dataset following an object-oriented approach:

- **Random Forest Classifier (RF)**
- **Feed-Forward Neural Network (FFNN)**
- **Convolutional Neural Network (CNN)**

All models follow a common interface pattern and are accessed through a unified `MnistClassifier` class, demonstrating proper OOP design principles.

## Project Structure
```
task1/
├── README.md                 # Project documentation
├── requirements.txt          # List of dependencies
├── MnistClassifier.ipynb     # Jupyter notebook with implementation and demo
```

## Solution Explanation

The solution implements a design pattern that emphasizes object-oriented programming principles with:

1. **Interface Definition**: `MnistClassifierInterface` abstract base class defines common methods all classifiers must implement:
   - `train(X_train, y_train)`: For training models on input data
   - `predict(X_test)`: For making predictions on new data

2. **Model Implementations**:
   - `RandomForestModel`: Uses scikit-learn's ensemble classifier
   - `FFNNModel`: Implements a feed-forward neural network using TensorFlow/Keras
   - `CNNModel`: Implements a convolutional neural network using TensorFlow/Keras

3. **Factory Pattern**: `MnistClassifier` class provides a unified interface to all models through parameter selection:
   ```python
   # Initialize any model using the same interface
   rf_model = MnistClassifier(algorithm='rf')  # Random Forest
   nn_model = MnistClassifier(algorithm='nn')  # Neural Network
   cnn_model = MnistClassifier(algorithm='cnn')  # CNN
   
   # Train and predict using identical methods
   rf_model.train(X_train, y_train)
   predictions = rf_model.predict(X_test)
   ```

## Model Architectures

### Random Forest
- Flattens 28x28 images into 784-dimensional vectors
- Uses 100 decision trees
- Optimized for MNIST classification

### Feed-Forward Neural Network
Architecture:
- Input layer (784 neurons)
- Hidden layer (128 neurons, ReLU)
- Dropout (0.2)
- Hidden layer (64 neurons, ReLU)
- Dropout (0.2)
- Output layer (10 neurons, softmax)

### Convolutional Neural Network
Architecture:
- Input layer (28×28×1)
- Conv2D (32 filters, 3×3) + ReLU
- MaxPooling (2×2)
- Conv2D (64 filters, 3×3) + ReLU
- MaxPooling (2×2)
- Flatten
- Dense (64 neurons, ReLU)
- Dropout (0.2)
- Output (10 neurons, softmax)

## Setting Up the Project

### Requirements
The project requires the following libraries:
- numpy
- tensorflow
- scikit-learn
- matplotlib
- seaborn

Install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Notebook
Open and run the Jupyter notebook to see the models in action:
```bash
jupyter notebook MnistClassifier.ipynb
```

## Performance Analysis
The models were evaluated on the MNIST test dataset (10,000 images):

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | 97.04% | ~54 seconds |
| FFNN | 97.56% | ~15 seconds |
| CNN | 99.07% | ~49 seconds |

## Edge Case Handling
The implementation includes robust error handling for:
- Invalid data types
- Empty datasets
- Incorrect input shapes
- Invalid algorithm names

These cases are tested in the notebook to ensure the models behave appropriately when presented with unexpected inputs.