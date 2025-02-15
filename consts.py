# URL датасета
DATASET_URL = "https://storage.yandexcloud.net/academy.ai/russian_literature.zip"

# URL весов Наташа
NAVEC_WEIGHTS_URL = "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar"

# Список писателей
CLASS_LIST=["Dostoevsky", "Tolstoy", "Turgenev", "Chekhov", "Lermontov", "Blok", "Pushkin", "Gogol", "Gorky", "Herzen", "Bryusov", "Nekrasov" ]

#5 писателей с самым большим кол-вом слов в датасете
CLASS_LIST_TOP_5=["Dostoevsky", "Tolstoy", "Turgenev", "Gorky", "Bryusov"]

# Количество слов, рассматриваемых как признаки
NUM_WORDS = 10000