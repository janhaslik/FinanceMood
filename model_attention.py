from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, MultiHeadAttention, GlobalAveragePooling1D
from keras._tf_keras.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import pandas as pd

data = pd.read_csv('data.csv', header=None, names=['Impact', 'Text'], encoding='ISO-8859-1')
X = data['Text'].str.lower()
y = data['Impact']


def prepare_inputs(data, max_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(data)
    X_transformed = tokenizer.texts_to_sequences(data)
    X_padded = pad_sequences(X_transformed, maxlen=max_len, padding='post')
    return X_padded, tokenizer


def prepare_outputs(data):
    encoder = LabelEncoder()
    encoder.fit(data)
    return encoder.transform(data), encoder


X, tokenizer = prepare_inputs(X)
y, encoder = prepare_outputs(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

num_classes = len(encoder.classes_)
vocab_size = len(tokenizer.word_index) + 1

input_text = Input(shape=(None,))
embedding_dim = 128
key_dim = 64
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_text)

attention = MultiHeadAttention(
    num_heads=4,
    key_dim=key_dim
)(embedding, embedding)
x = GlobalAveragePooling1D()(attention)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_text, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

EPOCHS = 25
history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=64)

eval_result = model.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)
