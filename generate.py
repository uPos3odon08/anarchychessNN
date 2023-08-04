import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute file path for the model
model_file_path = os.path.join(script_dir, 'reddit_post_titles_model.h5')

# Load the trained model
model = load_model(model_file_path)

# Load the max_sequence_length
max_sequence_length_path = os.path.join(script_dir, 'max_sequence_length.txt')
with open(max_sequence_length_path, 'r') as file:
    max_sequence_length = int(file.read())

# Load the tokenizer used during training
tokenizer_path = os.path.join(script_dir, 'tokenizer.pkl')
with open(tokenizer_path, 'rb') as file:
    tokenizer = pickle.load(file)

# Generate new Reddit post titles using the trained model
def generate_title(seed_text: str, next_words: int, model, max_sequence_length, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]
        predicted_index = np.argmax(predicted_probs) + 1  # Add 1 because the index starts from 1 in the tokenizer
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage:
seed_text = "What do i do in this position?"
next_words = 10
generated_title = generate_title(seed_text, next_words, model, max_sequence_length, tokenizer)
print("Generated Title:", generated_title)
