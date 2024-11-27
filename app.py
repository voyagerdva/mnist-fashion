from pickletools import optimize

import numpy as np
import tensorflow.keras.datasets as datasets
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.utils as utils
import tensorflow.keras.models
from tensorflow.python.keras.models import model_from_json

from tensorflow.python.keras.saving.saved_model.load import metrics

fashion_mnist = datasets.fashion_mnist
Sequential = models.Sequential
Dense = layers.Dense
to_categorical = utils.to_categorical
load_model = models.load_model




# Проверим, можно ли загрузить данные
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразование размерности изображений и нормализация данных:
x_train = x_train.reshape(60000, 784).astype("float32") / 255

# Преобразуем метки в категории:
y_train = utils.to_categorical(y_train, 10)

# Названия классов:
classes = ["футболка", "брюки", "свитер", "платье", "пальто", "туфли", "рубашка", "кроссовки", "сумка", "ботинки"]

# Создаем последовательную модель:
model = Sequential()

# Добавляем уровни сети:
model.add(Dense(800, input_dim=784, activation='relu'))
model.add(Dense(10, activation="softmax"))

# компилируем модель:
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())

# Обучаем сеть:
model.fit(
    x_train,
    y_train,
    batch_size=200,
    epochs=100,
    validation_split=0.2,
    verbose=1
)

model.save('fashion_mnist_dense.h5')
model = load_model('fashion_mnist_dense.h5')
model.save_weight('fashion_mnist_weights.h5')
model.load_weights('fashion_mnist_weights.h5')

json_string = model.to_json()
model = model_from_json(json_string)

# Запускаем сеть на входных данных
predictions = model.predict(x_test)

# Выводим один из результатов распознавания
print(predictions[0])

# Выводим номер класса, предсказанный нейросетью
print(np.argmax(y_train[0]))

