from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import pickle
import string  # Ensure this is imported
from copy import deepcopy
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
import os
from typing import Dict, List
from underthesea import word_tokenize
import underthesea
from keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Load models and necessary files
base_dir = os.path.dirname(__file__)

# for local host
# lstm_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_model.h5')
# lstm_cnn_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm_cnn.keras')

# for docker
lstm_model_path = os.path.join(os.path.dirname(__file__), 'lstm_model.h5')
lstm_cnn_model_path = os.path.join(os.path.dirname(__file__), 'lstm_cnn.h5')
print('LSTM path:', lstm_model_path)


lstm_model = load_model(lstm_model_path)
lstm_cnn_model = load_model(lstm_cnn_model_path)
roberta_model = RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment")
tokenizer_phobert = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

words_dict_path = os.path.join(base_dir, "utils", "words_dict.pkl")
with open(words_dict_path, "rb") as file:
    words = pickle.load(file)

tokenizer_file_path = os.path.join(base_dir, "utils", "tokenizer.pickle")
with open(tokenizer_file_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

DESIRED_SEQUENCE_LENGTH = 205

class TextRequest(BaseModel):
    text: str

def remove_punctuation(input_string):
    # Create a translation table that maps each punctuation character to None
    translation_table = str.maketrans("", "", string.punctuation)
    # Use translate to remove punctuation
    result_string = input_string.translate(translation_table)
    return result_string

def tokenize_vietnamese_sentence(sentence):
    return word_tokenize(remove_punctuation(sentence.lower()))

def feedbackSentimentAnalysis(result):
    if result == 0:
        return "Tiêu cực"
    elif result == 1:
        return "Trung tính"
    elif result == 2:
        return "Tích cực"

def sent2vec(message, word_dict=words):
    tokens = tokenize_vietnamese_sentence(message)
    vectors = []
    
    for token in tokens:
        if token not in word_dict.keys():
            continue
        token_vector = word_dict[token]
        vectors.append(token_vector)
    return np.array(vectors, dtype=float)

# def pad_sequences(X, desired_sequence_length=DESIRED_SEQUENCE_LENGTH):
#     X_copy = deepcopy(X)

#     for i, x in enumerate(X):
#         x_seq_len = x.shape[0]
#         sequence_length_difference = desired_sequence_length - x_seq_len
#         pad = np.zeros(shape=(sequence_length_difference, 200))
#         X_copy[i] = np.concatenate([x, pad])
  
#     return np.array(X_copy).astype(float)

def pad_sequence_sentence(sentence):
    array = sent2vec(sentence)
    arr_seq_len = array.shape[0]
    sequence_length_difference = DESIRED_SEQUENCE_LENGTH - arr_seq_len
        
    pad = np.zeros(shape=(sequence_length_difference, 200))
    array = np.array(np.concatenate([array, pad]))
    array = np.expand_dims(array, axis=0)
    return array

@app.post("/lstm_document")
def lstm_sentiment_analysis(request: TextRequest) -> Dict:
    # Extract the text from the request
    sentence = request.text
    array = pad_sequence_sentence(sentence)

    # Generate predictions
    predictions = lstm_model.predict(array)

    # Assuming predictions are probabilities and need to be converted to list
    predictions_list = predictions.tolist()
    
    print("Predictions LSTM: ", predictions_list)

    return {"predictions": predictions_list}

@app.post("/phobert_document")
def roberta_sentiment_analysis(request: TextRequest):
    # Tokenize and encode the input text
    tokenized_text = word_tokenize(request.text)
    input_ids = torch.tensor([tokenizer_phobert.encode(tokenized_text, add_special_tokens=True)])
    
    # Perform inference
    with torch.no_grad():
        out = roberta_model(input_ids)
        logits = out.logits.softmax(dim=-1).tolist()[0]  # Get probabilities for each class

    # Define sentiment labels
    sentiment_labels = ["Tiêu cực", "Tích cực", "Trung tính"]

    # Prepare response with sentiment probabilities
    response = {
        "sentiment": sentiment_labels[logits.index(max(logits))],  # Most probable sentiment
        "score": max(logits),  # Highest probability score
        "probabilities": {label: round(prob, 2) for label, prob in zip(sentiment_labels, logits)}  # Probabilities in percentage
    }
    print("Predictions PhoBert", response)

    return response

@app.post("/lstm_cnn_document")
def lstm_cnn_sentiment_analysis(request: TextRequest):
    text = request.text
    MAX_LEN = 200

    # Step 1: Text normalization
    normalized_text = underthesea.text_normalize(text)

    # Step 2: Word segmentation
    segmented_text = underthesea.word_tokenize(normalized_text)

    # Step 3: Convert to original format (as done during training)
    processed_text = ' '.join([sub.replace(' ', '_') for sub in segmented_text])

    # Step 4: Tokenization and padding
    tokenized_text = pad_sequences([tokenizer.texts_to_sequences([processed_text])[0]], maxlen=MAX_LEN)

    # Step 5: Predict using the trained model
    prediction = lstm_cnn_model.predict(tokenized_text)

    # Step 6: Determine predicted label
    predicted_label = np.argmax(prediction, axis=1)[0]

    # Step 7: Map predicted label to sentiment
    sentiment = feedbackSentimentAnalysis(predicted_label)
    print("Sentiment:", sentiment)

    # Step 8: Prepare response with sentiment and score
    response = {
        "sentiment": sentiment,
        "score": float(prediction[0][predicted_label]),
        "probabilities": {
            "Tiêu cực": round(float(prediction[0][0]), 2),
            "Trung tính": round(float(prediction[0][1]), 2),
            "Tích cực": round(float(prediction[0][2]), 2),
        }
    }
    print("Prediction LSTM CNN:", response)

    return response
# below code are api servers that function paragraph sentiment analysis

def pading_sequences(X, desired_sequence_length=DESIRED_SEQUENCE_LENGTH):
    X_copy = deepcopy(X)

    for i, x in enumerate(X):
        x_seq_len = x.shape[0]
        sequence_length_difference = desired_sequence_length - x_seq_len
        pad = np.zeros(shape=(sequence_length_difference, 200))
        X_copy[i] = np.concatenate([x, pad])
  
    return np.array(X_copy).astype(float)


@app.post("/lstm_paragraph")
def lstm_sentiment_analysis(request: List[TextRequest]):
    sentences = [req.text for req in request]
    vectors = [sent2vec(sentence) for sentence in sentences]
    sequences = pading_sequences(vectors)
    
    predictions = lstm_model.predict(sequences)
    predicted_labels = np.argmax(predictions, axis=1)
    sentiments = [feedbackSentimentAnalysis(label) for label in predicted_labels]
    
    results = [{"sentiment": sentiment, "score": float(predictions[i][label])} for i, (sentiment, label) in enumerate(zip(sentiments, predicted_labels))]
    
    return results

@app.post("/lstm_cnn_paragraph")
def lstm_cnn_sentiment_analysis(request: List[TextRequest]):
    sentences = [req.text for req in request]
    MAX_LEN = 200
    # Step 1: Text normalization
    normalized_sentences = [underthesea.text_normalize(sentence) for sentence in sentences]

    # Step 2: Word segmentation
    segmented_sentences = [underthesea.word_tokenize(sentence) for sentence in normalized_sentences]

    # Step 3: Convert to original format (as done during training)
    processed_sentences = [' '.join([sub.replace(' ', '_') for sub in sentence]) for sentence in segmented_sentences]

    # Step 4: Tokenization and padding
    tokenized_sentences = pad_sequences(tokenizer.texts_to_sequences(processed_sentences), maxlen=MAX_LEN)

    # Step 5: Predict using the trained model
    predictions = lstm_cnn_model.predict(tokenized_sentences)

    # Step 6: Determine predicted labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Step 7: Map predicted labels to sentiment (assuming feedbackSentimentAnalysis does this)
    sentiments = [feedbackSentimentAnalysis(label) for label in predicted_labels]

    # Step 8: Prepare results with sentiment and score
    results = [{"sentiment": sentiment, "score": float(predictions[i][label])} 
               for i, (sentiment, label) in enumerate(zip(sentiments, predicted_labels))]
    
    return results

roberta_tokenizer = AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)
@app.post("/phobert_paragraph")
def roberta_sentiment_analysis(request: List[TextRequest]):
    sentences = [req.text for req in request]
    results = []
    
    for sentence in sentences:
        tokenized_text = word_tokenize(sentence)
        input_ids = torch.tensor([roberta_tokenizer.encode(tokenized_text)])
        
        with torch.no_grad():
            out = roberta_model(input_ids)
            logits = out.logits.softmax(dim=-1).tolist()[0]
        
        sentiment = ["Tiêu cực", "Tích cực", "Trung tính"]
        max_index = logits.index(max(logits))
        
        results.append({"sentiment": sentiment[max_index], "score": logits[max_index]})
    
    return results

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Use this for debugging purposes only
    # logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
