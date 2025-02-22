import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERInference:
    def __init__(self, model_path):
        """
        Initialize NER inference class
        Args:
            model_path: Path to the trained model
        """
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
    def predict(self, text):
        """
        Predict entities in text
        Args:
            text: Input text string
        Returns:
            List of detected entities with their positions
        """
        # Tokenize input
        words = text.split()
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
        
        # Process predictions
        word_ids = inputs.word_ids()
        current_word = None
        results = []
        
        for idx, (word_id, pred) in enumerate(zip(word_ids, predictions)):
            if word_id != current_word:
                if pred == 1:  # ANIMAL label
                    results.append({
                        'entity': words[word_id],
                        'position': word_id
                    })
                current_word = word_id
                
        return results

def main():
    """Main inference function"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    model_path = os.path.join(base_dir, "models", "ner_model", "final")
    
    # Initialize inference
    inferencer = NERInference(model_path)
    
    # Run interactive testing
    print("\nEnter sentences to test (type 'quit' to exit):")
    print("\nFor example: You can see a cat here")
    while True:
        text = input("\nEnter a sentence: ")
        if text.lower() == 'quit':
            break
        
        results = inferencer.predict(text)
        if results:
            print("Detected animals:")
            for r in results:
                print(f"- {r['entity']} (Position: {r['position']})")
        else:
            print("No animals detected")

if __name__ == "__main__":
    main()