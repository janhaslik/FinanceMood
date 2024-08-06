import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, Flatten, Dense, Dropout
from keras._tf_keras.keras.regularizers import l2

data = pd.read_csv('data.csv', header=None, names=['Impact', 'Text'], encoding='ISO-8859-1')
X = data['Text']
y = data['Impact']

def prepare_inputs(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    tokenized_data = tokenizer.texts_to_sequences(data)
    return pad_sequences(tokenized_data), tokenizer

def prepare_outputs(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    return encoder.transform(data), encoder

X, tokenizer = prepare_inputs(X)
y, encoder = prepare_outputs(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 64
num_classes = len(encoder.classes_)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

EPOCHS = 100
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.summary()

results = model.evaluate(X_test, y_test, verbose=2)
print("Test loss, test acc:", results)
