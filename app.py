

import streamlit as st
from PIL import Image 
from segment import process

st.title('Распознавание MNIST')

image_file = st.file_uploader('Загрузить изображение', type=['png', 'jpg']) # Добавляем загрузчик файлов

if not image_file is None: # Выполняем блок, если загружено изображение
    image = Image.open(image_file) # Открываем изображение
    results = process(image_file) # Обрабатываем изображение с помощью функции, реализованной в другом файле
    st.write('Цифра на картинке: ', results)
