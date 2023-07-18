#импорт необходимых библиотек
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from generator import Generator
from tensorflow import keras
from keras.layers import Dense, LSTM

# инициализация обучающей, тестовой выборки
gen = Generator()
(x_train, y_train), (x_test, y_test) = (gen.x_train, gen.y_train), (gen.x_test, gen.y_test)

# преобразование выходных данных
# в вектора длиной 10
y_train_vect = tf.convert_to_tensor(y_train, dtype=float, name='y_train')
y_test_vect = tf.convert_to_tensor(y_test, dtype=float, name='y_test')

# структура сети: рекуррентная сеть с обратными связями
# во входной слой подаются вектора длиной 10
# слой будет зависить от формата входа (return_sequences=True)
# скрытый слой содержит 100 нейронов
# выходной слой содержит 1 простой нейрон - для выхода
model = keras.Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=[10, 1]))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
print(model.summary())

# компиляция сети: оптимизация по adam,
# в качестве метрики - отклонение от
# выходного значения тестовой выборки
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(x_train, y_train_vect, batch_size=1, epochs=500, verbose=True)

# вывод графика изменения ошибок
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()

# Оценим модель на тестовых данных, используя "evaluate"
model.evaluate(x_test, y_test_vect)

# тестирование модели
n = random.randint(0, 29)
x = np.expand_dims(x_test[n], axis=0)

print(f'Прогноз для:\n {x}')

res = model.predict(x, verbose=False)

print(f'Результат прогноза - {res}')
print(f'Реальный курс на этот день: {y_test_vect[n]}')