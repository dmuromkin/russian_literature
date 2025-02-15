from keras import layers, models, callbacks
from navec import Navec
import numpy as np
from consts import CLASS_LIST_TOP_5, NUM_WORDS
from tools import seq_vectorize, show_confusion_matrix, show_plot


WIN_SIZE = 1000   # Ширина окна в токенах
WIN_STEP = 100    # Шаг окна в токенах


def create_Embedding_model(word_index):
    
    navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

    # Загрузка весов
    embeddings_index = navec
    embedding_dim = 300    # размерность векторов эмбединга (300d в имени эмбединга)
    embedding_matrix = np.zeros((NUM_WORDS, embedding_dim))


    for word, i in word_index.items():
        if i < NUM_WORDS:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    print(f'Количество ненулевых строк в embedding_matrix: {np.count_nonzero(np.any(embedding_matrix, axis=1))}/{NUM_WORDS}')

             
    model = models.Sequential()
    model.add(layers.Embedding(NUM_WORDS, embedding_dim, input_length=WIN_SIZE, weights=embedding_matrix))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(40, activation="relu"))
    model.add(layers.Dropout(0.6))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(len(CLASS_LIST_TOP_5), activation='softmax'))

    model.layers[0].trainable = False
    
    return model


def train_model(train_seq: list, model: models.Sequential):
    
    WIN_SIZE = 1000   # Ширина окна в токенах
    WIN_STEP = 100    # Шаг окна в токенах
    EPOCHS = 10
    BATCH_SIZE = 64
    
    x_train, y_train, x_val, y_val, x_test, y_test = seq_vectorize(train_seq, 0.1, 0.1, CLASS_LIST_TOP_5, WIN_SIZE, WIN_STEP)
    
    CALLBACKS = [
        callbacks.ModelCheckpoint(filepath = 'best_model_pretrain.keras',
                                monitor = 'val_accuracy',
                                save_best_only = True,
                                mode = 'max',
                                verbose = 0)
    ]
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val), callbacks=CALLBACKS)
    model.save_weights('pre_trained_model.weights.h5') # можно сохранять не только модели, но и веса

    y_pred = model.predict(x_test)
    show_confusion_matrix(y_test, y_pred, CLASS_LIST_TOP_5)
    show_plot(history)