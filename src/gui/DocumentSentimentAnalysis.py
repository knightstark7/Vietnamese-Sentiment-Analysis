import streamlit as st
from PIL import Image
import pickle
from underthesea import word_tokenize
import numpy as np
import requests
from typing import Dict
import os
from visualization.visualization import draw_pie_chart_document

# for docker API
LSTM_API_URL = "http://api:8001/lstm_document"
PHOBERT_API_URL = "http://api:8001/phobert_document"
LSTM_CNN_API_URL = "http://api:8001/lstm_cnn_document"

# for local API
# LSTM_API_URL = "http://localhost:8001/lstm_document" 
# PHOBERT_API_URL = "http://localhost:8001/phobert_document"
# LSTM_CNN_API_URL = "http://localhost:8001/lstm_cnn_document"

words_dict_path = os.path.join(os.path.dirname(__file__), 'utils', 'words_dict.pkl')
absolute_path = os.path.abspath(words_dict_path)
print("Absolute Path:", absolute_path)
with open(words_dict_path, "rb") as file:
    print(file)
    words = pickle.load(file)

DESIRED_SEQUENCE_LENGTH = 205

def remove_punctuation(input_string):
    import string
    translation_table = str.maketrans("", "", string.punctuation)
    return input_string.translate(translation_table)

def tokenize_vietnamese_sentence(sentence):
    return word_tokenize(remove_punctuation(sentence.lower()))

def sent2vec(message, word_dict=words):
    tokens = tokenize_vietnamese_sentence(message)
    vectors = []
    
    for token in tokens:
        if token not in word_dict.keys():
            continue
        token_vector = word_dict[token]
        vectors.append(token_vector)
    return np.array(vectors, dtype=float)

def pad_sequence_sentence(sentence):
    array = sent2vec(sentence)
    arr_seq_len = array.shape[0]
    sequence_length_difference = DESIRED_SEQUENCE_LENGTH - arr_seq_len
        
    pad = np.zeros(shape=(sequence_length_difference, 200))
    array = np.concatenate([array, pad])
    array = np.expand_dims(array, axis=0)
    return array


def get_lstm_cnn_sentiment(sentence):
    response = requests.post(LSTM_CNN_API_URL, json={"text": sentence})
    if response.status_code == 200:
        return response.json()
    else:
        return {"sentiment": "Error", "score": 0, "probabilities": {"Tiêu cực": 0, "Tích cực": 0, "Trung tính": 0}}


def get_lstm_sentiment(text: str) -> Dict:
    response = requests.post(LSTM_API_URL, json={"text": text})
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) or isinstance(result, dict):
            return result
        else:
            return {"error": "Unexpected response format"}
    else:
        return {"error": "Error occurred"}

def get_phobert_sentiment(text: str) -> Dict:
    response = requests.post(PHOBERT_API_URL, json={"text": text})
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, dict):
            return result
        else:
            return {"error": "Unexpected response format"}
    else:
        return {"error": "Error occurred"}
    
def getSentiments(userText: str, model_type: str):
    if model_type == "LSTM":
        result = get_lstm_sentiment(userText)
    elif model_type == "PhoBERT":
        result = get_phobert_sentiment(userText)
    elif model_type == "LSTM_CNN":
        result = get_lstm_cnn_sentiment(userText)
    else:
        st.error("Invalid model type selected.")
        return

    if isinstance(result, dict) and "error" not in result:
        if model_type == "PhoBERT" or model_type == "LSTM_CNN":
            sentiment = result['sentiment']
            probabilities = result['probabilities']
            # Convert probabilities to a list in the order of sentiment labels
            sentiment_labels = ['Tiêu cực', 'Trung tính', 'Tích cực']
            percentages = [probabilities.get(label, 0) for label in sentiment_labels]
        else:
            percentages = result.get('predictions', [])[0]
            sentiment_labels = ['Tiêu cực', 'Trung tính', 'Tích cực']
            sentiment_index = np.argmax(percentages)
            sentiment = sentiment_labels[sentiment_index]

        # Determine the sentiment status and corresponding image
        if sentiment == "Tích cực":
            img_path = os.path.join(os.path.dirname(__file__), 'images', 'forapp', 'positive.png')
            image = Image.open(img_path)
        elif sentiment == "Tiêu cực":
            img_path = os.path.join(os.path.dirname(__file__), 'images', 'forapp', 'negative.png')
            image = Image.open(img_path)
        else:
            img_path = os.path.join(os.path.dirname(__file__), 'images', 'forapp', 'neutral.png')
            image = Image.open(img_path)

        # Display the results
        col1, col2 = st.columns(2)
        col1.image(image, caption=sentiment)
        
        pie_chart = draw_pie_chart_document(percentages)
        col2.plotly_chart(pie_chart, use_container_width=True)  # Use Plotly's plotly_chart
    else:
        st.error("Error: Unable to retrieve sentiment analysis.")


def renderPage():
    st.markdown(
        """
        <div style="text-align: center;">
            <h1>Document Feedback Sentiment Analysis</h1>
            <p>Analyze feedback from documents and classify sentiments as Positive, Negative, or Neutral.</p>
            <p>Upload a text file (.txt) or enter paragraphs directly for analysis.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    userText = st.text_input('User Input', placeholder='Enter text HERE')
    
    # Added LSTM_CNN to the model selection
    model_type = st.selectbox("Select model", ["LSTM", "PhoBERT", "LSTM_CNN"])
    
    # Button to analyze sentiment for input text
    if st.button('Analyze Sentiment'):
        if userText != "" and model_type is not None:
            st.components.v1.html("""
                <h3 style="color: #0284c7; 
                            font-family: Source Sans Pro, sans-serif; 
                            font-size: 28px; 
                            margin-bottom: 10px; 
                            margin-top: 50px;">
                    Result
                </h3>
            """, height=100)
            getSentiments(userText, model_type)

    # File upload for text file
    st.text("")
    st.subheader("Upload a Text File")
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

    if uploaded_file is not None:
        # Read the content of the uploaded file
        content = uploaded_file.read().decode("utf-8")
        # Split content into lines for analysis
        # sentences = content.splitlines()
        
        st.text_area("File Content:", value=content, height=200, max_chars=None)

        if st.button('Analyze Uploaded File'):
            if content and model_type is not None:
                st.components.v1.html("""
                    <h3 style="color: #0284c7; 
                                font-family: Source Sans Pro, sans-serif; 
                                font-size: 28px; 
                                margin-bottom: 10px; 
                                margin-top: 50px;">
                        File Analysis Result
                    </h3>
                """, height=100)
                getSentiments(content, model_type)
