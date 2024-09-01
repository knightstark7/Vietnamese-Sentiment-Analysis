import streamlit as st
import pandas as pd
import requests
import re
from underthesea import word_tokenize
import string
from visualization.visualization import draw_pie_chart_paragraph

# API docker 
LSTM_API_URL = "http://api:8001/lstm_paragraph"
PHOBERT_API_URL = "http://api:8001/phobert_paragraph"
LSTM_CNN_API_URL = "http://api:8001/lstm_cnn_paragraph"

# API endpoint for LSTM sentiment analysis
# LSTM_API_URL = "http://localhost:8001/lstm_paragraph"
# PHOBERT_API_URL = "http://localhost:8001/phobert_paragraph"
# LSTM_CNN_API_URL = "http://localhost:8001/lstm_cnn_paragraph"

def remove_punctuation(input_string):
    # Create a translation table that maps each punctuation character to None
    translation_table = str.maketrans("", "", string.punctuation)

    # Use translate to remove punctuation
    result_string = input_string.translate(translation_table)

    return result_string

def count_tokens(sentence):
    return len(word_tokenize(remove_punctuation(sentence.lower())))

def get_model_sentiment(sentences, model):
    if model == "LSTM":
        response = requests.post(LSTM_API_URL, json=[{"text": sentence} for sentence in sentences])
    elif model == "PhoBERT":
        response = requests.post(PHOBERT_API_URL, json=[{"text": sentence} for sentence in sentences])
    elif model == "LSTM_CNN":
        response = requests.post(LSTM_CNN_API_URL, json=[{"text": sentence} for sentence in sentences])
    else:
        return [{"sentiment": "Error", "score": 0} for _ in sentences]

    if response.status_code == 200:
        return response.json()
    else:
        return [{"sentiment": "Error", "score": 0} for _ in sentences] 

def process_paragraph_input(paragraph, model):
    sentences = re.split(r'[.!?]', paragraph)  # Split paragraph into sentences by '.!?'
    cumulative_sentences = []
    current_text = ""
    token_counts = []

    for sentence in sentences:
        if sentence.strip():  # Check if the sentence is not empty after stripping whitespace
            current_text += sentence.strip() + ". "  # Add the sentence to the cumulative text
            cumulative_sentences.append(current_text.strip())
            token_counts.append(count_tokens(current_text.strip()))

    # Perform sentiment analysis for each cumulative sentence
    results = get_model_sentiment(cumulative_sentences, model)

    # Prepare dataframe for display
    df = pd.DataFrame({
        "text": cumulative_sentences,
        "tokens": token_counts,
        "sentiment": [result["sentiment"] for result in results],
        "score": [result["score"] for result in results]
    })

    st.subheader("Paragraph Sentiment Analysis:")
    st.dataframe(df, width=800)

    return df



def process_and_display_data(uploaded_file, model):
    sentences = uploaded_file.read().decode("utf-8").splitlines()

    # Perform sentiment analysis for all sentences at once
    results = get_model_sentiment(sentences, model)

    # Create a DataFrame for displaying the results
    df = pd.DataFrame({
        "feedback": sentences,
        "sentiment": [result["sentiment"] for result in results],
        "score": [result["score"] for result in results]
    })

    # Calculate sentiment distribution
    sentiment_counts = df["sentiment"].value_counts()
    sentiments = sentiment_counts.index
    counts = sentiment_counts.values
    percentages = counts / counts.sum()

    # Create two columns in Streamlit layout with specified width ratio
    col1, col2 = st.columns([7, 3])  # 70% and 30% width ratio

    # Display the DataFrame in the first column
    with col1:
        st.subheader("Predictions:")
        st.dataframe(df.sample(min(50, len(df))), width=800)

    # Display the pie chart in the second column
    with col2:
        fig = draw_pie_chart_paragraph(percentages, sentiments)
        st.plotly_chart(fig, use_container_width=True)

def renderPage():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Paragraph and Sentence Sentiment Analysis</h1>
            <p>This tool analyzes the sentiment of feedback provided in paragraphs and sentences, classifying them as Positive, Negative, or Neutral.</p>
            <p>Upload a text file (.txt) or enter paragraphs directly for analysis.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )


    # Model selection
    model = st.selectbox("Select model:", ["LSTM", "PhoBERT", "LSTM_CNN"])

    # Text Area for Paragraph Input
    paragraph_input = st.text_area("Enter a paragraph for sentiment analysis:")

    if st.button("Analyze"):
        if paragraph_input:
            with st.spinner("Analyzing..."):
                process_paragraph_input(paragraph_input, model)

    # File Upload
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file:
        process_and_display_data(uploaded_file, model)


