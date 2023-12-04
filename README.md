# RuBert VK Classifier

[Русская версия](./README.md)
|
[English version](./README.md)



## Contents

1. [Structure and model classes](README.md#structure-and-model-classes)

2. [Description of the class CustomDataset](README.md#description-of-the-class-customdataset)

    2.1 [\_\_init__()](README.md#\_\_init__)

    2.2 [\_\_len__()](README.md#\_\_len__)

    2.3 [\_\_getitem__()](README.md#\_\_getitem__)

3. [Description of the class BertClassifier](README.md#description-of-the-class-bertclassifier)

    3.1 [\_\_init__()](README.md#\_\_init__)

    3.2 [predict()](README.md#predict)

    3.3 [fit()](README.md#fit)

    3.4 [eval()](README.md#eval)

4. [Model training](README.md#model-training)

    4.1 [Load data for training and validation](README.md#load-data-for-training-and-validation)

    4.2 [Model training](README.md#model-training)

    4.3 [Model evaluation](README.md#model-evaluation)

    4.4 [Saving the model](README.md#saving-the-model)

5. [Using the trained model](README.md#using-the-trained-model)

    5.1 [Load the model](README.md#load-the-model)

    5.2 [Label prediction](README.md#label-prediction)

## Structure and model classes

The main model training logic is located in the file [train.ipynb](./train.ipynb)

Example usage: [work.ipynb](./work.ipynb)

The model allows to determine the expected group of directions by the student's public communities preferences in VK.

The file [bert_dataset.py](./bert_dataset.py) contains the main logic (class) of storing and normalizing the dataset during training.

The file [bert_classifier.py](./bert_classifier.py) contains the class, aimed at working with the model directly (training, prediction).

In folder [trained_models](./trained_models) you can find pretrained models.

Folder [reports](./reports) contains reports of the trained model after using it on test dataset.

Folder [dataset](./dataset) contains the dataset files in `.csv` format.

Folder [output](./output) contains the output trained model files in `.pt` format.

## [CustomDataset](./bert_dataset.py) explanation

### \_\_init__()

An initialization method for a class. It takes in parameters for texts, targets, tokenizer, and max_len. It assigns the parameter values to instance variables with the same names.

### \_\_len__()

A special method in a class. It returns the length of the texts attribute of an

### \_\_getitem__()

A method for a dataset class. It retrieves an item from the dataset at the given index. It takes the index as an argument and returns a dictionary containing various properties of the item, such as the original text, input token IDs, attention mask, and target values. The method uses a tokenizer to encode the text and prepares it for further processing.

## [BertClassifier](./bert_classifier.py) explanation

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

### load_model()

This code defines a function called load_model that loads a model from a specified path. The function returns the loaded model.



## Model training

The complete code for the model training example can be found in the [train.ipynb](./train.ipynb) file.

The following steps are required to train the model:

* Load the training and validation data.
* Initialize the model.
* Train the model.
* Evaluate the model.
* Save the model.

### Load data for training and validation

You can use any data loading environment to load data. For example, you can use the pandas library to load data from a CSV file.
(The [train.csv](./dataset/train.csv) and [validate.csv](./dataset/validate.csv) files will be used for this example.)

```python
import pandas as pd

train_data = pd.read_csv('train.csv')
validate_data = pd.read_csv('validate.csv')
```

### Initializing the model

The [BertClassifier](./bert_classifier.py) class is used to initialize the model.

``` ``python
from bert_classifier import BertClassifier

classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny', # Path to the repository with the model to be trained
    tokenizer_path='cointegrated/rubert-tiny', # Path to tokenizer
    n_classes=41, # Number of classes to train
    epochs=60, # Number of training epochs
    max_len=512, # Maximum text size
    model_save_path='./output/model.pt' # Path to save the model
)
```
Accepted as arguments are:

* model_path - path to the repository with the model to be retrained
* tokenizer_path - path to tokenizer
* n_classes - number of classes to be trained
* epochs - number of training epochs
* max_len - maximum text size
* model_save_path - path to save the model

```python
classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=41,
    epochs=60,
    max_len=512,
    model_save_path='./output/model.pt'.
)
```

Then we connect the model to the training and validation data.

``` ``python
classifier.preparation(
    X_train=list(train_data['groups']), # Training fields of the table with text
    y_train=list(train_data['code']), # Training table fields with actual group codes
    X_valid=list(valid_data['groups']), # Validating fields of the table with text
    y_valid=list(valid_data['code']), # Validating table fields with real group codes
)
```

Accepted as arguments are:

* X_train - training fields of the table with text
* y_train - training fields of the table with real group codes
* X_valid - validating fields of the table with text
* y_valid - validating table fields with real group codes

### Training the model

The [train()](./bert_classifier.py) method is used to train the model.

```python
classifier.train()
```

### Model evaluation

The `precision_recall_fscore_support()` function from the `sklearn.metrics` library is used to estimate the model.

`` ``python
from sklearn.metrics import precision_recall_fscore_support

test_data = pd.read_csv('./dataset/test.csv')
labels = list(test_data['code'])

predictions = [classifier.predict(t) for t in texts]

precision, recall, f1score = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)[:3]
```

We get three metrics:

* precision
* recall - completeness
* f1score - F-measure

````python
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'f1score: {f1score}')
```

### Saving the model

The model is automatically saved to the `output` folder.

## Using the trained model

You can see the full code for an example of how to use the model in the file [work.ipynb](./work.ipynb).

To use the model, the following steps must be performed:

* Get the text to be checked. (Get it in any convenient way)
* Load the model.
* Predict the label for the received text.

### Load the model

```python
from bert_classifier import BertClassifier

classifier = BertClassifier(
    model_path='cointegrated/rubert-tiny',
    tokenizer_path='cointegrated/rubert-tiny',
    n_classes=41,
    epochs=60,
    max_len=512,
    model_save_path='./output/model.pt'.
)
```

The constructor accepts as arguments:

* model_path - path to the repository with the source model
* tokenizer_path - path to tokenizer
* n_classes - number of classes for training
* epochs - number of training epochs
* max_len - maximum text size
* model_save_path - path to the saved model

### Label prediction

The [predict()](./bert_classifier.py) method is used for label prediction.

In our case, the data for validation will be taken from table [test.csv](./dataset/test.csv) using pandas library.

```python
import pandas as pd

test_data = pd.read_csv('./dataset/test.csv')
texts = list(test_data['groups'])

predictions = [classifier.predict(t) for t in texts]
```

Output the predictions.

````python
print(predictions)
```