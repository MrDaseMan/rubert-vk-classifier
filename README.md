# RuBert VK Classifier

## EN

The main logic is located in the file [main.ipynb](./main.ipynb)

The model allows to determine the expected group of directions by the student's preferences in VK.

The file [bert_dataset.py](./bert_dataset.py) contains the main logic (class) of storing and normalizing the dataset during training.

The file [bert_classifier.py](./bert_classifier.py) contains the class, aimed at working with the model directly (training, prediction).

## RU

Вся основная логика находится в файле [main.ipynb](./main.ipynb)

Модель позволяет определять предполагаемую группу направлений поступления студента по его предпочтениям в VK.

Файл [bert_dataset.py](./bert_dataset.py) содержит основную логику (класс) хранения и нормализации датасета на этапе обучения.

Файл [bert_classifier.py](./bert_classifier.py) содержит класс, нацеленный на работу с непосредственно моделью (обучение, предсказание).