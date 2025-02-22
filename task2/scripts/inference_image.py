import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras import layers, models


class ImageClassifier:
    def __init__(self, model_path, class_names=None):
        """
        Initialize the image classifier.
        
        Args:
            model_path (str): Path to the saved model
            class_names (list): List of class names. If None, uses default classes.
        """
        self.model_path = model_path
        self.class_names = class_names or [
            'butterfly', 'cat', 'chicken', 'cow', 'dog',
            'elephant', 'horse', 'sheep', 'spider', 'squirrel'
        ]
        self.model = self._load_model()

    def _create_model(self):
        """Create the model architecture."""
        base_model = ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        return model

    def _load_model(self):
        """Load the trained model."""
        try:
            # Create new model with same architecture
            new_model = self._create_model()
            
            # Enable unsafe deserialization and load the old model
            tf.keras.config.enable_unsafe_deserialization()
            old_model = tf.keras.models.load_model(self.model_path)
            
            # Copy weights from old model to new model
            # Skip the first layer (Lambda) when copying weights
            for new_layer, old_layer in zip(new_model.layers, old_model.layers[1:]):
                new_layer.set_weights(old_layer.get_weights())
                
            return new_model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, image_path, return_top_k=3):
        """
        Predict the class of an image.
        
        Args:
            image_path (str): Path to the image file
            return_top_k (int): Number of top predictions to return
            
        Returns:
            tuple: (top_prediction, prediction_scores)
                - top_prediction: string name of the most likely class
                - prediction_scores: list of (class_name, score) tuples for top k predictions
        """
        try:
            # Load and preprocess image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top k predictions
            top_k_idx = np.argsort(predictions[0])[-return_top_k:][::-1]
            top_predictions = [
                (self.class_names[idx], float(predictions[0][idx]))
                for idx in top_k_idx
            ]
            
            return self.class_names[np.argmax(predictions[0])], top_predictions
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")


def main():
    # Define default paths relative to the script location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts folder
    TASK2_DIR = os.path.dirname(SCRIPT_DIR)  # task2 folder
    MODEL_PATH = os.path.join(TASK2_DIR, "models", "animal_classifier_final.keras")
    IMAGE_PATH = os.path.join(TASK2_DIR, "test_img1.jpg")

    # Print status
    print("\nStarting image classification...")
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Looking for test image at: {IMAGE_PATH}")

    # Validate paths
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    print("\n✅ Found all required files!")

    # Initialize classifier and make prediction
    classifier = ImageClassifier(MODEL_PATH)
    top_class, predictions = classifier.predict(IMAGE_PATH)

    # Print results
    print(f"\nPredicted class: {top_class}")
    print("\nTop 3 predictions:")
    for class_name, score in predictions:
        print(f"{class_name}: {score*100:.2f}%")


if __name__ == "__main__":
    main()