import os
import requests
import zipfile
from io import BytesIO
import glob # Вспомогательный модуль для работы с файловой системой
from consts import CLASS_LIST, CLASS_LIST_TOP_5, DATASET_URL, NAVEC_WEIGHTS_URL
from model import create_Embedding_model, train_model
from tools import balance_dataset, show_sample_distribution, tokenize_texts

def load_dataset(url, dataset_path="dataset"):
    
    if not os.path.exists(dataset_path):
        print(f"Папка {dataset_path} не найдена. Загружаем архив...")
        
        response = requests.get(url)
        if response.status_code == 200:
            os.makedirs(dataset_path, exist_ok=True)
            
            with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            
            print(f"Успешно!")
        else:
            print("Ошибка загрузки архива. Проверьте ссылку.")
            

def load_Navec_archive(url, dataset_path="navec_hudlit_v1_12B_500K_300d_100q.tar"):
    
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} не найден. Загружаем архив...")
        
        response = requests.get(url)
        if response.status_code == 200:            
            with open("navec_hudlit_v1_12B_500K_300d_100q.tar", "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):  # Загружаем файл частями
                    file.write(chunk)
        else:
            print("Ошибка загрузки архива. Проверьте ссылку.")


def read_dataset():
    all_texts = {} # Собираем в словарь весь датасет

    for author in CLASS_LIST:
        all_texts[author] = '' # Инициализируем пустой строкой новый ключ словаря
        for path in glob.glob('./dataset/prose/{}/*.txt'.format(author)) +  glob.glob('./dataset/poems/{}/*.txt'.format(author)): # Поиск файлов по шаблону
            with open(f'{path}', 'r', encoding="utf8", errors='ignore') as f: # игнорируем ошибки (например символы из другой кодировки)
                # Загрузка содержимого файла в строку
                text = f.read()         
            all_texts[author]  += ' ' + text.replace('\n', ' ') # Заменяем символ перехода на новую строку пробелом
    
    total = sum(len(i) for i in all_texts.values())
    print(f'Датасет состоит из {total} символов')
    
    return all_texts


def main():
    load_dataset(DATASET_URL)
    load_Navec_archive(NAVEC_WEIGHTS_URL)
    
    all_texts = read_dataset()
    #show_sample_distribution(all_texts.values(), CLASS_LIST)
    
    train_seq, word_index = tokenize_texts(all_texts)
    train_seq_balanced, CLASS_LIST_BALANCED = balance_dataset(train_seq)
    
    #################################### Топ 5 писателей #################################################
    train_seq_top_5 = []

    for author in CLASS_LIST_BALANCED:
        if author in CLASS_LIST_TOP_5:
            cls = CLASS_LIST_BALANCED.index(author)  
            train_seq_top_5.append(train_seq_balanced[cls])
            
    model = create_Embedding_model(word_index)
    train_model(train_seq_top_5, model)

if __name__ == "__main__":
    main()