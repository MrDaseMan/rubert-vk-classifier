# RuBert VK Classifier

[Русская версия](./README.ru.md)
| [English version](./README.md)

## Содержание
1. [Структура и классы модели](README.ru.md#структура-и-классы-модели)

2. [Описание класса CustomDataset](README.ru.md#описание-класса-customdataset)

    2.1 [\_\_init__()](README.ru.md#\_\_init__)

    2.2 [\_\_len__()](README.ru.md#\_\_len__)

    2.3 [\_\_getitem__()](README.ru.md#\_\_getitem__)

3. [Описание класса BertClassifier](README.ru.md#описание-класса-bertclassifier)

    3.1 [\_\_init__()](README.ru.md#\_\_init__)

    3.2 [predict()](README.ru.md#predict)

    3.3 [fit()](README.ru.md#fit)

    3.4 [eval()](README.ru.md#eval)

4. [Обучение модели](README.ru.md#обучение-модели)

    4.1 [Загрузка данных для обучения и проверки](README.ru.md#загрузка-данных-для-обучения-и-проверки)

    4.2 [Обучение модели](README.ru.md#обучение-модели)

    4.3 [Оценка модели](README.ru.md#оценка-модели)

    4.4 [Сохранение модели](README.ru.md#сохранение-модели)

5. [Использование обученной модели](README.ru.md#использование-обученной-модели)

    5.1 [Загрузка модели](README.ru.md#загрузка-модели)

    5.2 [Предсказание метки](README.ru.md#предсказание-метки)

## Структура и классы модели

Вся основная логика обучения модели находится в файле [train.ipynb](./train.ipynb)

Пример использования обученной модели: [work.ipynb](./work.ipynb)

Модель позволяет определять предполагаемую группу направлений поступления студента по его предпочтениям в выборе публичных сообществ в VK.

Файл [bert_dataset.py](./bert_dataset.py) содержит основную логику (класс) хранения и нормализации датасета на этапе обучения.

Файл [bert_classifier.py](./bert_classifier.py) содержит класс, нацеленный на работу с непосредственно моделью (обучение, предсказание).

В папке [trained_models](./trained_models) можно найти предобученные модели.

Папка [reports](./reports) содержит отчеты о процессе использования модели на этапе тестирования.

Папка [dataset](./dataset) содержит датасеты в формате `.csv`.

Папка [output](./output) содержит файлы модели после обучения в формате `.pt`.

### Описание класса [CustomDataset](./bert_dataset.py)

#### \_\_init__()

Метод инициализации класса. Он принимает параметры texts, targets, tokenizer и max_len. Значения параметров присваиваются переменным экземпляра с теми же именами.

#### \_\_len__()

Специальный метод в классе. Он возвращает длину атрибута texts в классе

#### \_\_getitem__()

Метод для класса набора данных. Он извлекает элемент из набора данных по заданному индексу. В качестве аргумента принимается индекс и возвращается словарь, содержащий различные свойства элемента, такие как исходный текст, идентификаторы входных маркеров, маска внимания и целевые значения. Метод использует токенизатор для кодирования текста и подготовки его к дальнейшей обработке.


### Описание класса [BertClassifier](./bert_classifier.py)

Конструктор класса (__init__) инициализирует объект с различными параметрами. Он принимает пути к модели и токенизатору, а также имеет необязательные параметры для количества классов, количества эпох, максимальной длины текстов и пути для сохранения модели.

Внутри конструктора код загружает модель и токенизатор по заданным путям, устанавливает устройство - CUDA, если доступно, или CPU, задает максимальную длину текстов и количество эпох.

После этого он получает количество выходных признаков с определенного слоя модели и модифицирует классификатор модели так, чтобы в нем было заданное количество классов. Наконец, модель перемещается на указанное устройство.

#### predict()

Метод preparation инициализирует наборы данных и загрузчики данных для обучения и проверки. В качестве параметров он принимает данные для обучения и проверки, а также метки. Метод создает два набора данных (train_set и valid_set) с помощью пользовательского класса CustomDataset и два загрузчика данных (train_loader и valid_loader) с помощью класса DataLoader. Также инициализируются некоторые вспомогательные объекты, такие как оптимизатор (AdamW), планировщик (get_linear_schedule_with_warmup) и функция потерь (CrossEntropyLoss).

Возвращает объект, содержащий предсказанный класс и все вероятности прочих классов.

#### fit()

Метод fit() обучает модель по обучающим данным. Он выполняет итерации по обучающим данным, прямое и обратное распространение, обновление параметров модели и вычисление точности и потерь при обучении. Метод возвращает точность обучения и потери в виде кортежа.

#### eval()

Метод под названием eval оценивает модель на валидационном множестве и возвращает точность и потери. Он выполняет следующие действия:

* Переводит модель в режим оценки.
Инициализирует пустые списки для потерь и счетчик для правильных предсказаний.
* Итерирует загрузчик валидации, который предоставляет пакеты данных.
* Перемещает входные данные, маску внимания и цели на устройство (например, GPU).
* Пропускает входные данные через модель для получения выходных данных.
* Вычисляет предсказанные метки классов и потери, используя логиты (выход модели до применения softmax).
* Обновляет счетчик правильных предсказаний и добавляет значение потерь в список.
* Вычисляется точность путем деления количества правильных предсказаний на размер валидационного множества.
* Вычисляет среднее значение потерь из списка потерь.

Возвращает значения точности и потерь.

#### train()

Метод train обучает модель в течение заданного количества эпох. Он инициализирует переменную best_accuracy равной 0, а затем выполняет итерации по каждой эпохе.

В течение каждой эпохи вызывается метод fit для обучения модели и выводится значение потерь и точности обучения. Затем вызывается метод eval для оценки модели на валидационном множестве и выводится значение потерь и точности при валидации.

Если точность проверки превышает предыдущую наилучшую точность, то модель сохраняется и обновляется best_accuracy до нового значения.

Наконец, загружается лучшая модель и присваивается атрибуту self.model.

#### predict()

Этот код определяет функцию predict, которая принимает текстовый входной сигнал и возвращает предсказанную метку класса. Она использует токенизатор для кодирования входного текста, а затем передает закодированный текст модели для предсказания. Предсказанная метка класса определяется путем взятия argmax выходных логарифмов модели.

#### load_model()

Этот код определяет функцию load_model, которая загружает модель из указанного пути. Функция возвращает модель.

## Обучение модели

Полный код примера обучения модели можно найти в файле [train.ipynb](./train.ipynb).

Для обучения модели необходимо выполнить следующие действия:

* Загрузить данные для обучения и проверки.
* Инициализировать модель.
* Обучить модель.
* Оценить модель.
* Сохранить модель.

### Загрузка данных для обучения и проверки

Для загрузки данных можно использовать любую среду загрузки данных. Например, можно использовать библиотеку pandas для загрузки данных из CSV-файла.
(Для примера будут использованы файлы [train.csv](./dataset/train.csv) и [validate.csv](./dataset/validate.csv).)

```python
import pandas as pd

train_data = pd.read_csv('train.csv')
validate_data = pd.read_csv('validate.csv')
```

### Инициализация модели

Для инициализации модели используется класс [BertClassifier](./bert_classifier.py).

```python
from bert_classifier import BertClassifier

classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny', # Путь до репозитория с дообучаемой моделью
    tokenizer_path='cointegrated/rubert-tiny', # Путь до токенизатора
    n_classes=41, # Количество классов для обучения
    epochs=60, # Количество эпох обучения
    max_len=512, # Максимальный размер текста
    model_save_path='./output/model.pt' # Путь до сохранения модели
)
```
В качестве аргументов принимаются:

* model_path - путь до репозитория с дообучаемой моделью
* tokenizer_path - путь до токенизатора
* n_classes - количество классов для обучения
* epochs - количество эпох обучения
* max_len - максимальный размер текста
* model_save_path - путь до сохранения модели

```python
classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=41,
    epochs=60,
    max_len=512,
    model_save_path='./output/model.pt'
)
```

Затем мы подключаем модель к обучающим и валидационным данным.

```python
classifier.preparation(
    X_train=list(train_data['groups']), # Обучающие поля таблицы с текстом
    y_train=list(train_data['code']), # Обучающие поля таблицы с реальными кодами групп
    X_valid=list(valid_data['groups']), # Валидирующие поля таблицы с текстом
    y_valid=list(valid_data['code'])  # Валидирующие поля таблицы с реальными кодами групп
)
```

В качестве аргументов принимаются:

* X_train - обучающие поля таблицы с текстом
* y_train - обучающие поля таблицы с реальными кодами групп
* X_valid - валидирующие поля таблицы с текстом
* y_valid - валидирующие поля таблицы с реальными кодами групп

### Обучение модели

Для обучения модели используется метод [train()](./bert_classifier.py).

```python
classifier.train()
```

### Оценка модели

Для оценки модели используется функция `precision_recall_fscore_support()` из библиотеки `sklearn.metrics`.

```python
from sklearn.metrics import precision_recall_fscore_support

test_data  = pd.read_csv('./dataset/test.csv')
labels = list(test_data['code'])

predictions = [classifier.predict(t) for t in texts]

precision, recall, f1score = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)[:3]
```

Получим три метрики:

* precision - точность
* recall - полнота
* f1score - F-мера

```python
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1score: {f1score}')
```

### Сохранение модели

Модель автоматически сохраняется в папку `output`.

## Использование обученной модели

Полный код примера использования модели можно посмотреть в файле [work.ipynb](./work.ipynb).

Для использования модели необходимо выполнить следующие действия:

* Получить текст для проверки. (Получаем любым удобным способом)
* Загрузить модель.
* Предсказать метку для полученного текста.

### Загрузка модели

```python
from bert_classifier import BertClassifier

classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=41,
    epochs=60,
    max_len=512,
    model_save_path='./output/model.pt'
)
```

В качестве аргументов конструктора принимаются:

* model_path - путь до репозитория с исходной моделью
* tokenizer_path - путь до токенизатора
* n_classes - количество классов для обучения
* epochs - количество эпох обучения
* max_len - максимальный размер текста
* model_save_path - путь до сохраненённой модели

### Предсказание метки

Для предсказания метки используется метод [predict()](./bert_classifier.py).

В нашем случае данные для проверки будут взяты из таблицы [test.csv](./dataset/test.csv) с использованием библиотеки pandas.

```python
import pandas as pd

test_data  = pd.read_csv('./dataset/test.csv')
texts = list(test_data['groups'])

predictions = [classifier.predict(t) for t in texts]
```

Выводим предсказания.

```python
print(predictions)
```