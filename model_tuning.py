import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Dropout, Embedding, SpatialDropout1D, MultiHeadAttention, \
    GlobalAveragePooling1D, LSTM
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2

import keras_tuner as kt

data = pd.read_csv('data.csv', header=None, names=['Impact', 'Text'], encoding='ISO-8859-1')
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
X = data['Text'].str.lower()
y = data['Impact']


def prepare_inputs(data, max_words=len(set(X)), max_len=100):
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


def build_model(hp):
    input_text = Input(shape=(None,))
    embedding = Embedding(input_dim=vocab_size,
                          output_dim=hp.Int('embedding_dim', min_value=50, max_value=300, step=50))(input_text)
    dropout_emb = SpatialDropout1D(hp.Float('dropout_emb', min_value=0.3, max_value=0.5, step=0.1))(embedding)

    lstm_out = LSTM(hp.Int('lstm_units', min_value=128, max_value=256, step=64), return_sequences=True,
                    dropout=hp.Float('dropout_lstm', min_value=0.3, max_value=0.5, step=0.1))(dropout_emb)

    attention = MultiHeadAttention(
        num_heads=hp.Int('num_heads', min_value=4, max_value=8, step=2),
        key_dim=hp.Int('key_dim', min_value=64, max_value=128, step=32)
    )(lstm_out, lstm_out)

    pooled_output = GlobalAveragePooling1D()(attention)

    dense_units = hp.Int('dense_units', min_value=128, max_value=256, step=64)
    dropout_ctx = Dropout(hp.Float('dropout', min_value=0.4, max_value=0.5, step=0.1))(pooled_output)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(hp.Float('l2_reg', min_value=0.001, max_value=0.005, step=0.001)))(dropout_ctx)

    model = Model(inputs=input_text, outputs=output)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model



tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=50, factor=3, directory='tuner',
                     project_name='tuner_fin_mood')

EPOCHS = 25
BATCH_SIZE = 64

checkpoint_cb = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')

tuner.search(X_train, y_train, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stopping, checkpoint_cb])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    callbacks=[checkpoint_cb, early_stopping])

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
hypermodel.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE,
               callbacks=[checkpoint_cb, early_stopping])

eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)
model.summary()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()
