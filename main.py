import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Step 1: Load and preprocess the data
with open('reddit_post_titles.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()

data = [title.strip() for title in data if title.strip()]

# Step 2: Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

total_words = len(tokenizer.word_index) + 1
input_sequences = tokenizer.texts_to_sequences(data)

# Step 3: Prepare the training data
input_sequences = np.array(input_sequences)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
sequence_length = X.shape[1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Step 4: Build the model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=sequence_length))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train the model
model.fit(X, y, epochs=50, batch_size=128)

# Step 6: Generate new titles
def generate_title(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        predicted_index = np.argmax(model.predict(token_list), axis=-1)
        predicted_word = tokenizer.index_word[predicted_index[0]]
        seed_text += " " + predicted_word
    return seed_text

# Example usage:
generated_title = generate_title("Artificial intelligence")
print(generated_title)
