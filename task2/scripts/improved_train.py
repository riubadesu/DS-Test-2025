import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Set dataset path
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
dataset_path = os.path.join(BASE_DIR, "task2", "dataset", "raw-img")

# Check if dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

print("✅ Dataset found:", dataset_path)

# Define model save path
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), "task2", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
print(MODEL_DIR)

# Data loading parameters
batch_size = 32
img_size = (224, 224)

# Create data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

# Load and prepare datasets
train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_dataset.class_names
print("Classes:", class_names)

# Enable prefetching for better performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



# Apply data augmentation to training dataset
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Load base model
base_model = ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze base model layers
base_model.trainable = False

# Build improved model architecture
model = models.Sequential([
    # Preprocessing layer - handles normalization
    layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(x)),
    
    # Base model
    base_model,
    
    # New classification head
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile model with improved settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=1e-6
)

# Model checkpoint to save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Initial training
print("Initial training phase...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=15,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Fine-tuning phase
print("\nFine-tuning phase...")
# Unfreeze some layers of the base model
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train for a few more epochs
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Save final model
model_path = os.path.join(MODEL_DIR, "animal_classifier_final.keras")
model.save(model_path)
print(f"✅ Model saved successfully at: {model_path}")

# Function for prediction
def predict_image(image_path, model, class_names):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    
    print("\nTop 3 predictions:")
    for idx in top_3_idx:
        print(f"{class_names[idx]}: {predictions[0][idx]*100:.2f}%")
    
    return class_names[np.argmax(predictions[0])]

# Test prediction visualization
def visualize_prediction(image_path, model, class_names):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    prediction = predict_image(image_path, model, class_names)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {prediction}")
    plt.show()

# Example usage for testing
if __name__ == "__main__":
    # Test image path
    test_image_path = os.path.join(BASE_DIR, "task2", "test_img2.jpeg")
    if os.path.exists(test_image_path):
        visualize_prediction(test_image_path, model, class_names)
    else:
        print("Test image not found!")