import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Constants
IMG_SIZE = (150, 150)
CLASS_NAMES = ['class0', 'class1', 'class2', 'class3', 'class4', 'class5']  # Replace with your actual class names in order

@st.cache_resource
def load_trained_model():
    model = load_model("best_model_small.h5")
    return model

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image) / 255.0  # Rescale as in training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def main():
    st.title("Multi-class Weather Image Classifier")
    st.write("Upload an image to classify the weather type.")

    model = load_trained_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds)
        confidence = preds[0][predicted_index] * 100

        st.write(f"**Prediction:** {CLASS_NAMES[predicted_index]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.write("### All Class Probabilities:")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {preds[0][i]*100:.2f}%")

if __name__ == "__main__":
    main()
