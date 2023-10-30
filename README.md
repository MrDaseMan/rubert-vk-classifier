# RuBert VK Classifier

# EN

The main logic is located in the file [main.ipynb](./main.ipynb)

The model allows to determine the expected group of directions by the student's preferences in VK.

The file [bert_dataset.py](./bert_dataset.py) contains the main logic (class) of storing and normalizing the dataset during training.

The file [bert_classifier.py](./bert_classifier.py) contains the class, aimed at working with the model directly (training, prediction).

In folder [trained_models](./trained_models) you can find pretrained models.

Folder [reports](./reports) contains reports of the trained model after using it on test dataset.

Folder [dataset](./dataset) contains the dataset files in `.csv` format.

Folder [output](./output) contains the output trained model files in `.pt` format.

## [CustomDataset](./bert_dataset) explanation

### \_\_init__()

An initialization method for a class. It takes in parameters for texts, targets, tokenizer, and max_len. It assigns the parameter values to instance variables with the same names.

### \_\_len__()

A special method in a class. It returns the length of the texts attribute of an

## \_\_getitem__()

A method for a dataset class. It retrieves an item from the dataset at the given index. It takes the index as an argument and returns a dictionary containing various properties of the item, such as the original text, input token IDs, attention mask, and target values. The method uses a tokenizer to encode the text and prepares it for further processing.

## [BertClassifier](./bert_classifier) explanation

### \_\_init__()

A class constructor (__init__) initializes an object with various parameters. It takes in the paths to a model and a tokenizer, and also has optional parameters for the number of classes, number of epochs, maximum length of texts, and the path to save the model.

Inside the constructor, the code loads the model and tokenizer from the given paths, sets the device to either CUDA if available or CPU, sets the maximum length of the texts, and sets the number of epochs.

It then gets the number of output features from a specific layer of the model, and modifies the classifier of the model to have the specified number of classes. Finally, it moves the model to the specified device.

### predict()

A method called preparation initializes datasets and data loaders for training and validation. It takes in training and validation data and labels as parameters. The method creates two datasets (train_set and valid_set) using a custom dataset class called CustomDataset, and two data loaders (train_loader and valid_loader) using the DataLoader class. It also initializes some helper objects such as an optimizer (AdamW), a scheduler (get_linear_schedule_with_warmup), and a loss function (CrossEntropyLoss).

### fit()

A method fit() trains a model using the training data. It iterates over the training data, performs forward and backward propagation, updates the model parameters, and calculates the training accuracy and loss. The method returns the training accuracy and loss as a tuple.

### eval()

A method called eval evaluates a model on a validation set and returns the accuracy and loss. It does the following:

* Sets the model to evaluation mode.
Initializes empty lists for losses and a counter for correct predictions.
* Iterates over the validation loader, which provides batches of data.
* Moves the input data, attention mask, and targets to the device (e.g., GPU).
* Passes the input data through the model to get the outputs.
* Calculates the predicted class labels and the loss using the logits (output of the model before applying softmax).
* Updates the counter for correct predictions and appends the loss value to the list.
* Calculates the accuracy by dividing the number of correct predictions by the size of the validation set.
* Computes the average loss from the list of losses.

Returns the accuracy and loss.

### train()

A train method trains a model for a specified number of epochs. It initializes a variable best_accuracy to 0 and then iterates over each epoch.

Within each epoch, it calls the fit method to train the model and prints the training loss and accuracy. It then calls the eval method to evaluate the model on a validation set and prints the validation loss and accuracy.

If the validation accuracy is greater than the previous best accuracy, it saves the model and updates best_accuracy to the new value.

Finally, it loads the best model and assigns it to the self.model attribute.

### predict()

This code defines a predict function that takes a text input and returns the predicted class label. It uses a tokenizer to encode the input text, and then passes the encoded input to a model for prediction. The predicted class label is determined by taking the argmax of the model's output logits.

# RU

Вся основная логика находится в файле [main.ipynb](./main.ipynb)

Модель позволяет определять предполагаемую группу направлений поступления студента по его предпочтениям в VK.

Файл [bert_dataset.py](./bert_dataset.py) содержит основную логику (класс) хранения и нормализации датасета на этапе обучения.

Файл [bert_classifier.py](./bert_classifier.py) содержит класс, нацеленный на работу с непосредственно моделью (обучение, предсказание).

В папке [trained_models](./trained_models) можно найти предобученные модели.

Папка [reports](./reports) содержит отчеты о процессе использования модели на этапе тестирования.

Папка [dataset](./dataset) содержит датасеты в формате `.csv`.

Папка [output](./output) содержит файлы модели после обучения в формате `.pt`.

## Описание класса [CustomDataset](./bert_dataset)

### \_\_init__()

Метод инициализации класса. Он принимает параметры texts, targets, tokenizer и max_len. Значения параметров присваиваются переменным экземпляра с теми же именами.

### \_\_len__()

Специальный метод в классе. Он возвращает длину атрибута texts в классе

## \_\_getitem__()

Метод для класса набора данных. Он извлекает элемент из набора данных по заданному индексу. В качестве аргумента принимается индекс и возвращается словарь, содержащий различные свойства элемента, такие как исходный текст, идентификаторы входных маркеров, маска внимания и целевые значения. Метод использует токенизатор для кодирования текста и подготовки его к дальнейшей обработке.


## Описание класса [BertClassifier](./bert_classifier)

Конструктор класса (__init__) инициализирует объект с различными параметрами. Он принимает пути к модели и токенизатору, а также имеет необязательные параметры для количества классов, количества эпох, максимальной длины текстов и пути для сохранения модели.

Внутри конструктора код загружает модель и токенизатор по заданным путям, устанавливает устройство - CUDA, если доступно, или CPU, задает максимальную длину текстов и количество эпох.

После этого он получает количество выходных признаков с определенного слоя модели и модифицирует классификатор модели так, чтобы в нем было заданное количество классов. Наконец, модель перемещается на указанное устройство.

### predict()

Метод preparation инициализирует наборы данных и загрузчики данных для обучения и проверки. В качестве параметров он принимает данные для обучения и проверки, а также метки. Метод создает два набора данных (train_set и valid_set) с помощью пользовательского класса CustomDataset и два загрузчика данных (train_loader и valid_loader) с помощью класса DataLoader. Также инициализируются некоторые вспомогательные объекты, такие как оптимизатор (AdamW), планировщик (get_linear_schedule_with_warmup) и функция потерь (CrossEntropyLoss).

### fit()

Метод fit() обучает модель по обучающим данным. Он выполняет итерации по обучающим данным, прямое и обратное распространение, обновление параметров модели и вычисление точности и потерь при обучении. Метод возвращает точность обучения и потери в виде кортежа.

### eval()

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

### train()

Метод train обучает модель в течение заданного количества эпох. Он инициализирует переменную best_accuracy равной 0, а затем выполняет итерации по каждой эпохе.

В течение каждой эпохи вызывается метод fit для обучения модели и выводится значение потерь и точности обучения. Затем вызывается метод eval для оценки модели на валидационном множестве и выводится значение потерь и точности при валидации.

Если точность проверки превышает предыдущую наилучшую точность, то модель сохраняется и обновляется best_accuracy до нового значения.

Наконец, загружается лучшая модель и присваивается атрибуту self.model.

### predict()

Этот код определяет функцию predict, которая принимает текстовый входной сигнал и возвращает предсказанную метку класса. Она использует токенизатор для кодирования входного текста, а затем передает закодированный текст модели для предсказания. Предсказанная метка класса определяется путем взятия argmax выходных логарифмов модели.