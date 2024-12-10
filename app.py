import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model

sentiment_pipeline = pipeline("text-classification", model="./sentiment_model", tokenizer="./sentiment_model")

# Streamlit App

st.title("Sentiment Analysis App")
user_input = st.text_area("Enter text:", "")

if st.button("Analyze"):
    if user_input.strip():
        result = sentiment_pipeline(user_input)
        st.write(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.2f}")
    else:
        st.write("Please enter some text to analyze")