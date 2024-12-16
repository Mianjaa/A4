import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

# Load IMDB dataset
def load_imdb_dataset(container_path='./aclImdb'):
    movie_reviews = load_files(container_path)
    X, y = movie_reviews.data, movie_reviews.target
    return [text.decode('utf-8') for text in X], y

def word_embedding_mlp():
    # Load dataset
    X, y = load_imdb_dataset()
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tokenization
    MAX_WORDS = 10000
    MAX_LEN = 200
    
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Padding
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    
    # Build MLP with Embedding
    model = Sequential([
        Embedding(MAX_WORDS, 32, input_length=MAX_LEN),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train_pad, y_train, 
              epochs=5, 
              batch_size=32, 
              validation_split=0.2,
              verbose=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f"\nWord Embedding + MLP Results:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    word_embedding_mlp()
