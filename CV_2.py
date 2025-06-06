from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

DATASET_PATH = "dataset/"

# Определение числа классов и режима классификации
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def preprocess_image(image_path, target_size=(128, 128)):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0

        # Normalize with ImageNet mean and std (optional, improves accuracy if model was trained similarly)
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - imagenet_mean) / imagenet_std

        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def predict_image(relative_path):
    image_path = os.path.join(DATASET_PATH, relative_path)
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        return

    # Проверка, что файл — это корректное изображение
    try:
        img_check = Image.open(image_path)
        img_check.verify()
    except (OSError, IOError):
        print(f"Error: image has been damaged - {image_path}")
        return

    # Загрузка модели
    try:
        model = tf.keras.models.load_model("image_classifier.h5")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Предобработка изображения
    img = preprocess_image(image_path)
    if img is None:
        return

    # Предсказание
    prediction = model.predict(img)
    class_names = sorted(os.listdir(DATASET_PATH))

    if class_mode == "binary":
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
    else:
        predicted_class = class_names[np.argmax(prediction)]

    print(f"Model predicted: {predicted_class}")

    # Отображение изображения
    img_to_show = Image.open(image_path)
    plt.imshow(img_to_show)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Пример вызова
predict_image("dog/360_F_1037429549_gOOV5tZR4Hf8Gd5ZI8urXs0n9IvMjxxo.jpg")