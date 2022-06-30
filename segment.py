

from tensorflow.keras.models import load_model
MODEL_NAME =   'model_fmr_all.h5'
import numpy as np
from PIL import Image 
model = load_model(MODEL_NAME) # Загружаем веса
INPUT_SHAPE = (28, 28, 1)


def process(image_file):
    image = Image.open(image_file).convert('L') # Открываем обрабатываемый файл
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0])) # Изменяем размер изображения в соответствии со входом сети
    array = np.array(resized_image, dtype='float64') # Меняем размерность тензора для подачи в сеть
    array = array.reshape(-1, 28, 28, 1)
    cls_image = np.argmax(model.predict(array))

    return cls_image
