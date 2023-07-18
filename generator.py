# импорт необходимых библиотек
from pycbrf import ExchangeRates  # библиотека ЦБ РФ
from datetime import datetime
from datetime import timedelta
import numpy as np


# получение данных при помощи
# библиотеки запросов к сайту ЦБ РФ
# возвращает список из полученных данных
# delta - количество дней от тек. даты
# count - количество получаемых данных
def get_data(delta, count):
    start_date = datetime.now().date() - timedelta(days=delta)
    data_list = []

    for i in range(count):
        curr_date = start_date + timedelta(days=i)
        rates = ExchangeRates(str(curr_date))
        data_list.append(rates['EUR'].value)

    return data_list


# получение списка кортжей
# по 10 элементов на основе
# полученных данных
# со сдвигом на 1 каждый шаг
def get_rates(data_list, count):
    rates_list = []
    start_pos = 0

    for i in range(count):
        curr_rate = data_list[start_pos:(start_pos + 10):]
        rates_list.append(curr_rate)
        start_pos += 1

    return rates_list


# нормализация входных данных
def get_input(delta, count):
    return np.asarray(get_rates(delta, count)).astype(np.float32)


# нормализация выходных данных
def get_output(delta, count):
    count += 10
    return np.asarray(get_data(delta, count)[10:]).astype(np.float32)


# оболочка полученных данных
# в класс для удобства
class Generator:
    def __init__(self):
        self.x_train = get_input(100, 90)
        self.y_train = get_output(100, 90)

        self.x_test = get_input(40, 30)
        self.y_test = get_output(40, 30)
