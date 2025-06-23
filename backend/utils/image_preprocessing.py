import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image


class PreProcessor:
    
    def __init__(self, target_size: tuple =(224, 224)):
        self.target_size =target_size

        
    def preprocess_image(self, uploaded_image) -> np.ndarray:
        img = Image.open(uploaded_image).convert("RGB")
        img = img.resize(self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)