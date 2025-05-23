import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image

model = load_model('weather_model.keras')
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']  

st.title("Weather Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).resize((64, 64))
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    st.write("Prediction:", class_names[np.argmax(prediction)])
