import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class NERTester:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
    def predict(self, text):
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

def test_model():
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "ner_model", "final")
    
    # Initialize tester
    tester = NERTester(model_path)
    
    # Test cases that match your training data format
    test_cases = [
        "There is a dog in the picture",
        "I can see a beautiful butterfly in this image",
        "The cat is playing with a toy",
        "This is a picture of mountains and trees",
        "The elephant and giraffe are at the zoo",
        "A squirrel appears in the picture",
        "There are two horses in the field",
        "The spider built a web in the corner"
    ]
    
    print("\nTesting NER model with various examples:\n")
    print("-" * 50)
    
    for text in test_cases:
        print(f"\nInput: {text}")
        results = tester.predict(text)
        if results:
            print("Detected animals:")
            for r in results:
                print(f"- {r['entity']} (Position: {r['position']})")
        else:
            print("No animals detected")
    
    print("\n" + "-" * 50)
    
    # Interactive testing
    print("\nEnter your own sentences to test (type 'quit' to exit):")
    while True:
        text = input("\nEnter a sentence: ")
        if text.lower() == 'quit':
            break
        
        results = tester.predict(text)
        if results:
            print("Detected animals:")
            for r in results:
                print(f"- {r['entity']} (Position: {r['position']})")
        else:
            print("No animals detected")

if __name__ == "__main__":
    test_model()