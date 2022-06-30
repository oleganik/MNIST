

import streamlit as st
from PIL import Image 
from segment import process
from tensorflow.keras.models import load_model
MODEL_NAME =   'model_fmr_all.h5'
import numpy as np
from PIL import Image 
model = load_model(MODEL_NAME) # Загружаем веса
INPUT_SHAPE = (28, 28, 1)

def process(image_file):
    image = Image.open(image_file).convert('L') # Открываем обрабатываемый файл
    resized_image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0])) # Изменяем размер изображения в соответствии со входом сети
    array = np.array(resized_image, dtype='float64') / 255 # Меняем размерность тензора для подачи в сеть
    array = array.reshape(1, 28, 28, 1)
    pred = model.predict(array)
    cls_image = np.argmax(pred)

    return cls_image

st.title('Распознавание MIST')

image_file = st.file_uploader('Загрузить изображение', type=['png', 'jpg']) # Добавляем загрузчик файлов

if not image_file is None: # Выполняем блок, если загружено изображение
    image = Image.open(image_file) # Открываем изображение
    results = process(image_file) # Обрабатываем изображение с помощью функции, реализованной выше
    st.write('Цифра на картинке: ', results)
