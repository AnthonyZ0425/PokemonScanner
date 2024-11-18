import sqlite3
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
from io import BytesIO
from key import DB_PATH

# Constants
IMAGE_SIZE = (224, 224)  # Input size for EfficientNet
BATCH_SIZE = 32
EPOCHS = 10

# Fetch data from the SQLite database
def fetch_data_from_db():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    query = "SELECT name, series, setName, imageURL FROM cards"
    cursor.execute(query)
    data = cursor.fetchall()  # Returns a list of tuples 
    return data

# Download and preprocess images
def download_and_preprocess_image(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        img = load_img(BytesIO(response.content), target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None

# Prepare the dataset
def prepare_dataset(data):
    images = []
    labels = []

    # Create a mapping for unique classes (name + set_name)
    label_map = {}
    for name, series, setName, _ in data:
        label = f"{name}-{series}-{setName}"
        if label not in label_map:
            label_map[label] = len(label_map)  # Assign a unique index to each label

    for name, series, setName, imageURL in data:
        label = f"{name}-{series}-{setName}"
        label_index = label_map[label]

        # Download and preprocess the image
        img_array = download_and_preprocess_image(imageURL)
        if img_array is not None:
            images.append(img_array)
            labels.append(label_index)

        if not images or not labels:
            raise ValueError("No valid images or labels could be prepared. Check the data or image URLs.")


    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")
    return images, labels, label_map

# Create a simple CNN model
def create_model(input_shape, num_classes):
    """
    Creates a simple CNN model for image classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels)
        num_classes (int): Number of unique classes in the dataset
    
    Returns:
        TensorFlow model: CNN model for image classification
    """
    model = models.Sequential([ # Each layer performs specific operations on input data
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), # Convolutional layer with 32 filters
        layers.MaxPooling2D((2, 2)), # Max pooling layer
        layers.Conv2D(64, (3, 3), activation='relu'), # Convolutional layer with 64 filters
        layers.MaxPooling2D((2, 2)), # Max pooling layer
        layers.Conv2D(128, (3, 3), activation='relu'), # Convolutional layer with 128 filters
        layers.Flatten(), # Flatten the output to feed into a dense layer
        layers.Dense(128, activation='relu'), # Dense layer with 128 units
        layers.Dense(num_classes, activation='softmax') # Output layer with num_classes units and softmax activation
    ])
    return model

# Step 5: Train the model
def train_model(model, train_images, train_labels):
    """
    Trains the model with given training images and labels.

    Args:
        model: TensorFlow model to train
        train_images: Array of training images
        train_labels: Array of training labels
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train the model
    model.fit(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,  # Adjust the number of epochs
        validation_split=0.2  # Use 20% of the data for validation
    )

# Main execution
if __name__ == "__main__":
    try:
        print("Fetching data from the database...")
        data = fetch_data_from_db()

        print("Preparing dataset...")
        train_images, train_labels, label_map = prepare_dataset(data)

        print(f"Dataset prepared with {len(train_images)} samples and {len(label_map)} classes.")

        print("Creating model...")
        model = create_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=len(label_map))

        print("Training model...")
        train_model(model, train_images, train_labels)

        print("Model training complete. Saving model...")
        model.save("pokemon_card_recognition_model.h5")
        print("Model saved as 'pokemon_card_recognition_model.h5'.")
    except Exception as e:
        print(f"An error occurred: {e}")