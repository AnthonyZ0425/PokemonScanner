import tensorflow as tf
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from pokemontcgsdk import Card, RestClient
from key import API_KEY

RestClient.configure(API_KEY)

def load_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def predict_image(img):
    preprocessed_image = preprocess_image(img)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def fetch_card_info(card_id):
    card = Card.find(card_id)
    return {
        "name": card.name,
        "hp": card.hp,
        "rarity": card.rarity,
        "image_url": card.images['small']
    }
card_id = 'basep-1'  # Pokémon card ID (replace with actual ID)
card_info = fetch_card_info(card_id)
print(f"Card Name: {card_info['name']}")
print(f"HP: {card_info['hp']}")
print(f"Rarity: {card_info['rarity']}")
print(f"Image URL: {card_info['image_url']}")

# Load image and make prediction
image_url = card_info['image_url']
img = load_image(image_url)
predictions = predict_image(img)

# Display top predictions
for i, (imagenet_id, label, score) in enumerate(predictions):
    print(f"{i + 1}. {label}: {score * 100:.2f}%")