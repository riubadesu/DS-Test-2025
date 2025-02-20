import os
import json
import random
import pandas as pd

# Define dataset paths relative to the script's location
DATASET_PATH = os.path.abspath("../dataset/raw-img")
OUTPUT_PATH = os.path.abspath("../dataset/ner_training_data")

# Ensure OUTPUT_PATH exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get animal classes from dataset
animal_classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
print("Animal Classes:", animal_classes)

# Create training sentence templates
templates = [
    "There is a {animal} in the picture",
    "I can see a {animal} in this image",
    "The photo shows a {animal}",
    "This image contains a {animal}",
    "A {animal} appears in the picture",
    "The {animal} is clearly visible",
    "You can see a {animal} here",
    "The picture displays a {animal}",
    "A {animal} is shown in this photo",
    "This is a picture of a {animal}",
]

# Generate training data
def generate_training_examples(animal_classes, templates, num_examples_per_template=5):
    training_data = []
    for template in templates:
        for _ in range(num_examples_per_template):
            for animal in animal_classes:
                sentence = template.format(animal=animal.lower())
                start_idx = sentence.find(animal.lower())
                end_idx = start_idx + len(animal)

                annotation = {
                    "sentence": sentence,
                    "entities": [(start_idx, end_idx, "ANIMAL")]
                }
                training_data.append(annotation)
    return training_data

training_data = generate_training_examples(animal_classes, templates)

# Add negative examples (sentences without animals)
negative_templates = [
    "This is a beautiful landscape",
    "I can see trees and mountains",
    "The sky is very blue today",
    "There are clouds in this picture",
    "This shows a beautiful sunset",
]

for template in negative_templates:
    training_data.append({
        "sentence": template,
        "entities": []
    })

# Shuffle dataset
random.shuffle(training_data)

# Split dataset into train/validation (80/20 split)
split_idx = int(len(training_data) * 0.8)
train_data = training_data[:split_idx]
val_data = training_data[split_idx:]

# Save datasets
def save_dataset(data, filename):
    with open(os.path.join(OUTPUT_PATH, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

save_dataset(train_data, 'train.json')
save_dataset(val_data, 'val.json')

# Display dataset stats
print("\nDataset Statistics:")
print("Total examples: {len(training_data)}")
print("Training examples: {len(train_data)}")
print("Validation examples: {len(val_data)}")

# Save dataset as CSV for reference
df = pd.DataFrame(training_data)
df.to_csv(os.path.join(OUTPUT_PATH, 'dataset_overview.csv'), index=False)
print("Dataset saved successfully.")
