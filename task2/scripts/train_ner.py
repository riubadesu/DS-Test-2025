import os
import json
import random
import logging
import argparse
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, dataset_path, output_path):
        """
        Initialize data preparation class
        Args:
            dataset_path: Path to raw image dataset containing animal class folders
            output_path: Path to save processed NER training data
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Templates for generating training data
        self.templates = [
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
        
        self.negative_templates = [
            "This is a beautiful landscape",
            "I can see trees and mountains",
            "The sky is very blue today",
            "There are clouds in this picture",
            "This shows a beautiful sunset",
        ]

    def get_animal_classes(self):
        """Get list of animal classes from dataset directory"""
        return [d for d in os.listdir(self.dataset_path) 
                if os.path.isdir(os.path.join(self.dataset_path, d))]

    def generate_training_examples(self, animal_classes, num_examples_per_template=5):
        """Generate training examples using templates"""
        training_data = []
        
        # Generate positive examples
        for template in self.templates:
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
        
        # Add negative examples
        for template in self.negative_templates:
            training_data.append({
                "sentence": template,
                "entities": []
            })
            
        return training_data

    def prepare_dataset(self, train_ratio=0.8, num_examples_per_template=5):
        """Prepare and save the complete dataset"""
        animal_classes = self.get_animal_classes()
        logger.info(f"Found animal classes: {animal_classes}")
        
        # Generate and shuffle data
        training_data = self.generate_training_examples(
            animal_classes, 
            num_examples_per_template
        )
        random.shuffle(training_data)
        
        # Split dataset
        split_idx = int(len(training_data) * train_ratio)
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Save datasets
        self.save_dataset(train_data, 'train.json')
        self.save_dataset(val_data, 'val.json')
        
        # Save overview
        df = pd.DataFrame(training_data)
        df.to_csv(os.path.join(self.output_path, 'dataset_overview.csv'), index=False)
        
        logger.info(f"Total examples: {len(training_data)}")
        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Validation examples: {len(val_data)}")
        
        return train_data, val_data

    def save_dataset(self, data, filename):
        """Save dataset to JSON file"""
        filepath = os.path.join(self.output_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved dataset to {filepath}")

class NERTrainer:
    def __init__(
        self,
        model_name="bert-base-uncased",
        max_length=128,
        train_batch_size=16,
        eval_batch_size=16,
        learning_rate=5e-5,
        num_epochs=5
    ):
        """
        Initialize NER trainer
        Args:
            model_name: Name of the pretrained model to use
            max_length: Maximum sequence length
            train_batch_size: Training batch size
            eval_batch_size: Evaluation batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
        """
        self.model_name = model_name
        self.max_length = max_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Define label mappings
        self.label2id = {"O": 0, "ANIMAL": 1}
        self.id2label = {0: "O", 1: "ANIMAL"}
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            pad_to_multiple_of=8
        )

    def load_data(self, train_file: str, val_file: str):
        """Load and preprocess the data"""
        def process_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            processed_data = []
            for item in data:
                try:
                    processed = self.prepare_example(item["sentence"], item["entities"])
                    processed_data.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing example: {str(e)}")
                    continue
            
            return Dataset.from_dict({
                "tokens": [x["tokens"] for x in processed_data],
                "labels": [x["labels"] for x in processed_data]
            })
        
        train_dataset = process_file(train_file)
        val_dataset = process_file(val_file)
        
        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(val_dataset)} validation examples")
        
        return train_dataset, val_dataset

    def prepare_example(self, text: str, entities: list):
        """Prepare a single example with accurate token labeling"""
        words = text.split()
        labels = ["O"] * len(words)
        
        for start, end, label in entities:
            entity_positions = self.get_entity_positions(text, start, end)
            for pos in entity_positions:
                if pos < len(labels):
                    labels[pos] = "ANIMAL"
        
        return {
            "tokens": words,
            "labels": [self.label2id[label] for label in labels]
        }

    def get_entity_positions(self, text: str, start: int, end: int):
        """Get token positions for an entity"""
        words = text.split()
        char_count = 0
        entity_tokens = []
        
        for i, word in enumerate(words):
            word_start = char_count
            word_end = char_count + len(word)
            
            if i > 0:
                word_start += 1
                word_end += 1
                char_count += 1
            
            if word_end > start and word_start < end:
                entity_tokens.append(i)
            
            char_count += len(word)
        
        return entity_tokens

    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with tokens"""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length"
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        """Compute metrics for evaluation"""
        predictions = np.argmax(p.predictions, axis=2)
        
        true_labels = [[l for l in label if l != -100] for label in p.label_ids]
        true_predictions = [
            [p for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(predictions, p.label_ids)
        ]

        precision, recall, f1, _ = precision_recall_fscore_support(
            [l for labels in true_labels for l in labels],
            [p for preds in true_predictions for p in preds],
            average='binary',
            zero_division=0
        )
        
        accuracy = accuracy_score(
            [l for labels in true_labels for l in labels],
            [p for preds in true_predictions for p in preds]
        )

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, train_dataset, val_dataset, output_dir: str):
        """Train the model"""
        # Process datasets
        tokenized_train = train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_val = val_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        model_save_path = os.path.join(output_dir, "final")
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

def get_default_args():
    """Get default arguments for training"""
    args = argparse.Namespace()
    
    # Set default paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level from scripts folder
    
    # Data preparation arguments
    args.dataset_path = os.path.join(base_dir, 'dataset', 'raw-img')
    args.output_path = os.path.join(base_dir, 'dataset', 'ner_training_data')
    args.model_output_path = os.path.join(base_dir, 'models', 'ner_model')
    args.train_ratio = 0.8
    args.examples_per_template = 5
    
    # Model training arguments
    args.model_name = 'bert-base-uncased'
    args.max_length = 128
    args.train_batch_size = 16
    args.eval_batch_size = 16
    args.learning_rate = 5e-5
    args.num_epochs = 5
    
    return args

def main():
    """Main training function"""
    args = get_default_args()
    
    # Prepare data
    data_prep = DataPreparation(args.dataset_path, args.output_path)
    data_prep.prepare_dataset(
        train_ratio=args.train_ratio,
        num_examples_per_template=args.examples_per_template
    )
    
    # Initialize trainer
    trainer = NERTrainer(
        model_name=args.model_name,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # Load and train
    train_file = os.path.join(args.output_path, 'train.json')
    val_file = os.path.join(args.output_path, 'val.json')
    train_dataset, val_dataset = trainer.load_data(train_file, val_file)
    
    # Train model
    trainer.train(train_dataset, val_dataset, args.model_output_path)

if __name__ == "__main__":
    main()