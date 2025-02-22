## Getting Started
### 1. Clone the Repository
```bash
git clone https://github.com/riubadesu/DS-Test-2025.git
cd DS-Test-2025/task2
```

### 2. Install Dependencies
Run the following command to install all required Python libraries:
```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Models (Required before running the pipeline)
The pre-trained models are **not included** in this repository. Download them using the Kaggle API:

#### Animal Classifier Model
[Download from Kaggle](https://www.kaggle.com/datasets/liumar/animal-classifier)
```bash
kaggle datasets download -d liumar/animal-classifier -p models/
unzip models/animal-classifier.zip -d models/
```

#### NER Model
[Download from Kaggle](https://www.kaggle.com/datasets/liumar/ner-model)
```bash
kaggle datasets download -d liumar/ner-model -p models/
unzip models/ner-model.zip -d models/
```

If you are using `downloading.ipynb`, you can run it to automate these steps.

### 4. Run the Pipeline
You can test the pipeline using:
```bash
python scripts/pipeline.py "There is a cat in the picture." test_img1.jpg
```
This will return:
```
True (if the text matches the image)
False (if it does not)
```

## Project Structure
```
📂 task2/
 ├── README.md              # Project documentation
 ├── requirements.txt       # List of dependencies
 ├── scripts/               # Python scripts for training & inference
 │   ├── train_ner.py       # Train NER model
 │   ├── train_image.py     # Train Image Classification model
 │   ├── inference_ner.py   # NER inference
 │   ├── inference_image.py # Image classification inference
 │   ├── pipeline.py        # Final pipeline combining both models
 ├── test_img1.jpg          # Example test image
 ├── downloading.ipynb      # Dataset & model preparation
 ├── image_exploration.ipynb # Dataset analysis
```

## Dataset and Model Requirements

### For Pipeline Usage
To use the pipeline (`pipeline.py`) or run inferences (`inference_ner.py`, `inference_image.py`), you only need:
- Pre-trained models (download instructions in Getting Started section)
- No dataset required

### For Model Training and Analysis
The `downloading.ipynb` notebook handles the dataset preparation needed for:
- Training models from scratch (`train_ner.py`, `train_image.py`)
- Dataset exploration (`image_exploration.ipynb`)
- Model evaluation and validation

The notebook:
- Downloads the required animal classification dataset
- Performs necessary data preprocessing and translations
- Prepares the data structure for model training

Only run this step if you plan to train models from scratch or analyze the dataset.

## Detailed Script Documentation

### Named Entity Recognition (NER)

#### `train_ner.py`
This script serves two main purposes:
1. **Training Data Generation**:
   - Creates the `ner_data` dataset containing labeled sentences
   - Generates diverse examples of animal mentions in various contexts
   - Includes proper entity annotations for training

2. **Model Training**:
   - Utilizes a transformer-based architecture for NER
   - Trains on the generated dataset to recognize animal entities
   - Implements early stopping and model checkpointing
   - Saves the trained model for later use

#### `inference_ner.py`
The NER inference script provides an interface for entity extraction:
- Loads the pre-trained NER model
- Processes input text to identify animal mentions
- Returns detected entities with their positions in the text

Usage example:
```bash
python scripts/inference_ner.py
# Enter text when prompted:
# > "I saw a lion and a giraffe at the zoo"
# Output will show detected animals and their positions
```

### Image Classification

#### `train_image.py`
The image classification training script:
- Loads and preprocesses the animal image dataset
- Implements data augmentation techniques
- Trains a deep learning model for animal classification
- Includes validation steps and model evaluation
- Saves the best performing model

#### `inference_image.py`
This script handles image classification inference:
- Loads the trained classification model
- Preprocesses input images to match training requirements
- Returns predicted animal class and confidence score

Usage:
```bash
python scripts/inference_image.py
# Provide image path when prompted
```

### Pipeline Integration

#### `pipeline.py`
The main pipeline script combines both models:
1. Text Processing:
   - Takes user input text
   - Uses NER model to extract animal mentions
   - Handles various text formulations

2. Image Analysis:
   - Processes the input image
   - Performs classification
   - Returns predicted animal class

3. Comparison Logic:
   - Compares NER results with image classification
   - Returns boolean result based on match

The pipeline usage instructions are provided in the Getting Started section above.

## Model Training Requirements
If you choose to train the models from scratch:
1. Ensure all required datasets are downloaded via `downloading.ipynb`
2. Follow the training sequence:
   ```bash
   python scripts/train_ner.py  # Train NER model first
   python scripts/train_image.py  # Then train image classifier
   ```
3. Training configurations can be modified in the respective script files

## Technical Details
- NER Model: Transformer-based architecture optimized for entity recognition
- Image Classification: Deep learning model with support for 10+ animal classes
- Pipeline Integration: Efficient processing pipeline with minimal latency

Note: Pre-trained models are available for immediate use without training. 
Refer to the download instructions in the Getting Started section.

