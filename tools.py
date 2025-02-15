import keras
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from consts import CLASS_LIST, NUM_WORDS


def tokenize_texts(all_texts):
    # Используется встроенный в Keras токенизатор для разбиения текста и построения частотного словаря
    tokenizer = Tokenizer(num_words=NUM_WORDS,
                      filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',
                      lower=True, split=' ', char_level=False)

    # Построение частотного словаря по текстам
    tokenizer.fit_on_texts(all_texts.values())
    
    return tokenizer.texts_to_sequences(all_texts.values()), tokenizer.word_index

def balance_dataset(train_seq):
    mean_list = np.array([])
    for author in CLASS_LIST:
        cls = CLASS_LIST.index(author)
        mean_list = np.append(mean_list, len(train_seq[cls]))

    ################################## Балансировка датасета ####################################

    median = int(np.median(mean_list)) # медианное значение
    CLASS_LIST_BALANCED = [] # Сбалансированный набор меток
    seq_train_balanced = []
    for author in CLASS_LIST:
        cls = CLASS_LIST.index(author)
        if len(train_seq[cls]) > median * 0.6:
            seq_train_balanced.append(train_seq[cls][:median])
            CLASS_LIST_BALANCED.append(author)       
    
    #show_sample_distribution(seq_train_balanced, CLASS_LIST_BALANCED)
    
    return seq_train_balanced, CLASS_LIST_BALANCED

# Функция разбиения последовательности на отрезки скользящим окном
# Последовательность разбивается на части до последнего полного окна
# Параметры:
# sequence - последовательность токенов
# win_size - размер окна
# step - шаг окна
def seq_split(sequence, win_size, step):
    # Делим строку на отрезки с помощью генератора цикла
    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, step)]

################################ Функция векторизации последовательности и создания конечных выборок ##############################################
def seq_vectorize(
    seq_list,   # Последовательность
    test_split, # Доля на тестовую сборку
    val_split, # Доля на тестовую сборку
    class_list, # Список классов
    win_size,   # Ширина скользящего окна
    step        # Шаг скользящего окна
):

    # Списки для результирующих данных
    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    # Пробежимся по всем классам:
    for class_item in class_list:
        # Получим индекс класса
        cls = class_list.index(class_item)
        seq = seq_list[cls]

        # Пороговое значение индекса для разбивки на тестовую и обучающую выборки
        train_split = int(len(seq) * (1 - val_split - test_split))
        val_split_index = int(len(seq) * (1 - test_split))

        # Разбиваем последовательность токенов класса на отрезки
        vectors_train = seq_split(seq[:train_split], win_size, step)
        vectors_val = seq_split(seq[train_split:val_split_index], win_size, step)
        vectors_test = seq_split(seq[val_split_index:], win_size, step)

        # Добавляем отрезки в выборку
        x_train += vectors_train
        x_val += vectors_val
        x_test += vectors_test

        # Для всех отрезков класса добавляем метки класса в виде one-hot-encoding
        # Каждую метку берем len(vectors) раз, так она одинакова для всех выборок одного класса
        y_train += [keras.utils.to_categorical(cls, len(class_list))] * len(vectors_train)
        y_val += [keras.utils.to_categorical(cls, len(class_list))] * len(vectors_val)
        y_test += [keras.utils.to_categorical(cls, len(class_list))] * len(vectors_test)

    # Возвращаем результатов как numpy-массивов
    return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)

############################### Функция вывода графиков точности и ошибки #############################################
def show_plot(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
               label='График точности на обучающей выборке')
    ax1.plot(history.history['val_accuracy'],
               label='График точности на проверочной выборке')
    ax1.xaxis.get_major_locator().set_params(integer=True) # На оси х показываем целые числа
    ax1.set_xlabel('Эпоха обучения')
    ax1.set_ylabel('График точности')
    ax1.legend()

    ax2.plot(history.history['loss'],
               label='Ошибка на обучающей выборке')
    ax2.plot(history.history['val_loss'],
               label='Ошибка на проверочной выборке')
    ax2.xaxis.get_major_locator().set_params(integer=True) # На оси х показываем целые числа
    ax2.set_xlabel('Эпоха обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()
    plt.show()

############################### Функция вывода предсказанных значений  ###################################################
def show_confusion_matrix(y_true, y_pred, class_labels):
    # Матрица ошибок
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')
    # Округление значений матрицы ошибок
    cm = np.around(cm, 3)

    # Отрисовка матрицы ошибок
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'Матрица ошибок', fontsize=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()  # Убираем ненужную цветовую шкалу
    plt.xlabel('Предсказанные классы', fontsize=16)
    plt.ylabel('Верные классы', fontsize=16)
    fig.autofmt_xdate(rotation=45)          # Наклон меток горизонтальной оси
    plt.show()


    # Средняя точность распознавания определяется как среднее диагональных элементов матрицы ошибок
    print('\nСредняя точность распознавания: {:3.0f}%'.format(100. * cm.diagonal().mean()))
    

def show_sample_distribution(samples, classes):
    fig, ax = plt.subplots()
    ax.pie([len(i) for i in samples], # формируем список значений как длина символов текста каждого автора
        labels=classes,                    # список меток
        pctdistance=1.2,                      # дистанция размещения % (1 - граница окружности)
        labeldistance=1.4,                    # размещение меток (1 - граница окружности)
        autopct='%1.2f%%'                     # формат для % (2 знака после запятой)
        )
    plt.show()