import sys

import pandas as pd
import numpy as np

def task1():
    print("1.")
    print(np.ones((5,5)))

    print("\n2. "
          "Создаем массив [0, 10000) так как размер не указан в задании.")
    random_array = np.random.random(size=20)*10000
    print(random_array)

    print("\n3.")
    print(random_array.reshape(4, 5), "\n")

def task2():
    n = int(input("Введите n: "))
    m = int(input("Введите m: "))
    print("\nСоздаем массив [0, 10000) так как размер не указан в задании.")
    random_array = np.random.random((n,m))*10000
    print("Исходный массив:\n", random_array)

    print("\n1.")
    k = float(input("Введите k: "))
    print("Массив элементов, больших чем k: \n", random_array[random_array > k])

    print("\n2.")
    print("Математическое ожидание: ", np.mean(random_array))
    print("Дисперсия: ", np.var(random_array))
    print("Среднеквадратичное отклонение: ", np.std(random_array))

    print("\n3.")
    print("Минимальный элемент: ", np.min(random_array))
    print("Максимальный элемент: ", np.max(random_array), "\n")

def task3():
    print("Создаем два вектора размера 5 со значениями [0, 10000)")
    vector_a = np.random.random(5) * 10000
    vector_b = np.random.random(5) * 10000

    print("Вектор A: ", vector_a)
    print("Вектор B: ", vector_b)

    print("\n1.")
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a != 0 and norm_b != 0:
        cosine_similarity = dot_product / (norm_a * norm_b)
        print("Косинус угла между векторами: ", cosine_similarity)
    else:
        print("Один из векторов нулевой, угол не определен")

    print("\n2.")
    euclidean_distance = np.linalg.norm(vector_a - vector_b)
    print("Евклидово расстояние между векторами: ", euclidean_distance, "\n")

def task4():
    x_values = np.arange(-10, 10.1, 0.5)
    print("1. Создаем массив x значений:")
    print(x_values)

    print("\n2. Вычисляем значения функции y = x² - x + 2: ")
    y_values = x_values ** 2 - x_values + 2
    print(y_values)

    print("\n3. Создаем таблицу значений с использованием DataFrame:")
    table_df = pd.DataFrame({
        'x': x_values,
        'y': y_values
    })
    print(table_df.to_string(index=False), "\n")

def task5():
    print("1. Загрузка данных из файла sp500.csv")
    try:
        df_initial = pd.read_csv('sp500.csv', sep=',')
        print("Первые 10 строк данных:")
        print(df_initial.head(10))
    except FileNotFoundError:
        print("Ошибка: файл sp500.csv не найден в текущей директории")
        return

    print("\n2. Загрузка только выбранных столбцов")
    columns_to_load = ['Symbol', 'Name', 'Sector', 'Price', 'Book Value', '52 week low', '52 week high', 'Market Cap']
    df = pd.read_csv('sp500.csv', sep=',', usecols=columns_to_load, index_col='Symbol')
    print(df.head(10))

    print("\n3. Значения столбца Name:")
    print(df['Name'])

    print("\n4. Строка с индексом NFLX:")
    print(df.loc['NFLX'])

    print("\n5. Строка с номером 238 (по порядку):")
    print(df.iloc[237], "\n")

def task6():
    print("Загрузка данных из файла sp500.csv")
    columns_to_load = ['Symbol', 'Name', 'Sector', 'Price', 'Book Value', '52 week low', '52 week high', 'Market Cap']
    try:
        df = pd.read_csv('sp500.csv', usecols=columns_to_load, index_col='Symbol', sep=',')
    except FileNotFoundError:
        print("Ошибка: файл sp500.csv не найден в текущей директории")
        return

    print("\n1. Строки с 100 по 120:")
    print(df.iloc[99:120])

    print("\n2. Копия датафрейма без столбца Book Value:")
    df_copy = df.copy()
    df_copy = df_copy.drop('Book Value', axis=1)
    print(df_copy)

    print("\n3. Строки где 52 week low < 80:")
    low_filter = df[df['52 week low'] < 80]
    print(low_filter)

    print("\n4. Строки где Sector = 'Financials' или 'Energy':")
    sector_filter = df[df['Sector'].isin(['Financials', 'Energy'])]
    print(sector_filter)

    print("\n5. Строки где (Sector = 'Financials' или 'Energy') и 52 week low > 50:")
    complex_filter = df[(df['Sector'].isin(['Financials', 'Energy'])) & (df['52 week low'] > 50)]
    print(complex_filter, "\n")


def task7():
    print("Загрузка данных из файла tips.csv")
    try:
        df = pd.read_csv('tips.csv')
        print(df)
    except FileNotFoundError:
        print("Ошибка: файл tips.csv не найден в текущей директории")
        return

    print("\nБазовые статистики для столбца 'tip'")
    tips = df['tip']

    print("\n1. Количество значений:")
    print("   Количество значений: ", tips.count())

    print("\n2. Минимальная и максимальная цена:")
    print("   Минимальные чаевые: ", tips.min())
    print("   Максимальные чаевые: ", tips.max())

    print("\n3. Среднее значение, медиана и мода:")
    print("   Среднее значение: ", tips.mean())
    print("   Медиана: ", tips.median())
    print("   Мода: ", tips.mode().iloc[0])

    print("\n4. Дисперсия и среднеквадратичное отклонение:")
    print(f"   Дисперсия: ", tips.var())
    print(f"   Стандартное отклонение: ", tips.std())

    print("\n5. Ковариация и корреляция 'tip' с другими числовыми колонками:")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'tip']

    for col in numeric_cols:
        print("   С колонкой ", col, ":")
        print("     Ковариация: ", tips.cov(df[col]))
        print("     Корреляция: ", tips.corr(df[col]), "\n")

if __name__ == '__main__':
    print("Задание 1:")
    task1()

    print("Задание 2:")
    task2()

    print("Задание 3:")
    task3()

    print("Задание 4:")
    task4()

    print("Задание 5:")
    task5()

    print("Задание 6:")
    task6()

    print("Задание 7:")
    task7()
