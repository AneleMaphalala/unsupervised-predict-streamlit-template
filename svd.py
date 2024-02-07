import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

with open('resources\models\JL2_SVD.pkl', 'rb') as modelfile:
    model = pickle.load(modelfile)


def predict(text):
    # Preprocess text (if needed)
    # ...

    # Make prediction using your model
    prediction = model.predict([text])

    return prediction

def main():
    st.title('Your Streamlit App')
    st.sidebar.header('User Input')

    # Get user input
    user_input = st.sidebar.text_area('Enter text for prediction:', 'Type here...')

    # Make predictions when the user clicks the button
    if st.sidebar.button('Predict'):
        prediction = predict(user_input)
        st.success(f'Prediction: {prediction}')

if __name__ == '__main':
    main()