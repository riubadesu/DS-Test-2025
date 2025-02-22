#!/usr/bin/env python3
import os
import sys
import torch
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras import layers, models
import argparse
import keras

class AnimalPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.BASE_DIR = base_dir
        
        # Configure Keras
        keras.config.enable_unsafe_deserialization()
        
        # Load NER model
        self.ner_model_path = os.path.join(self.BASE_DIR, "models", "ner_model", "final")
        if not os.path.exists(self.ner_model_path):
            raise FileNotFoundError(f"NER model not found at {self.ner_model_path}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_model_path)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(self.ner_model_path)
        
        # Define class names for the image classifier
        self.class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
                           'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.image_model = self.create_and_load_image_model()

    def create_and_load_image_model(self):
        # Create model architecture using ResNet50 as base
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
        
        # Load trained weights
        model_path = os.path.join(self.BASE_DIR, "models", "animal_classifier_final.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Image classifier model not found at {model_path}")
            
        old_model = tf.keras.models.load_model(model_path)
        # Copy weights, skipping the Lambda layer
        for new_layer, old_layer in zip(model.layers, old_model.layers[1:]):
            new_layer.set_weights(old_layer.get_weights())
        
        return model

    def extract_animal_from_text(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Extract tokens labeled as ANIMAL
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        animal_tokens = []
        current_animal = []
        
        for token, pred in zip(tokens, predictions[0]):
            if pred == 1:  # ANIMAL label
                if token.startswith("##"):
                    current_animal.append(token[2:])
                else:
                    if current_animal:
                        animal_tokens.append("".join(current_animal))
                        current_animal = []
                    current_animal.append(token)
            else:
                if current_animal:
                    animal_tokens.append("".join(current_animal))
                    current_animal = []
        
        if current_animal:
            animal_tokens.append("".join(current_animal))
        
        # Clean up extracted animal names
        animals = [animal.lower() for animal in animal_tokens if animal not in ["[CLS]", "[SEP]", "[PAD]"]]
        return animals[0] if animals else None

    def classify_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get prediction
        predictions = self.image_model.predict(img_array, verbose=0)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        
        return predicted_class, confidence

    def process(self, text, image_path):
        """
        Process text and image to determine if they match.
        Returns:
            bool: True if the animal mentioned in text matches the image, False otherwise
        """
        try:
            # Extract animal from text
            text_animal = self.extract_animal_from_text(text)
            if not text_animal:
                return False
            
            # Classify image
            image_animal, _ = self.classify_image(image_path)
            
            # Return True if animals match
            return text_animal.lower() == image_animal.lower()
            
        except Exception as e:
            print(f"Error processing inputs: {str(e)}", file=sys.stderr)
            return False

def main():
    parser = argparse.ArgumentParser(description='Animal Recognition Pipeline')
    parser.add_argument('text', type=str, help='Text description of the animal')
    parser.add_argument('image', type=str, help='Path to the image file')
    parser.add_argument('--base_dir', type=str, help='Base directory containing models', default=None)
    
    args = parser.parse_args()
    
    try:
        pipeline = AnimalPipeline(args.base_dir)
        result = pipeline.process(args.text, args.image)
        print(result)
        return 0 if result else 1
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())