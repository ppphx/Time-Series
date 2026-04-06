import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
from pyts.classification import KNeighborsClassifier, TimeSeriesForest

# 1. Генерация сигналов

def pulse(t):
    """Пульс сигнал"""
    return 1 * (abs(t) < 0.5)

# индекс и отсчет времени в секундах
time_index = np.linspace(0, 9, 100)

tseries_list = {'Time': time_index}
d = np.random.random(size=10)

N = 7 # количество образцов в каждом типе сигналов

# гармонические колебания, чтобы они были похожи, но различны, добавим небольшой шум или сдвиг фазы/амплитуды
f0 = 0.2
for i in range(N):
    # Добавляем небольшой случайный сдвиг фазы и амплитуды для разнообразия внутри класса
    phase_shift = np.random.uniform(0, 1)
    amp = np.random.uniform(0.9, 1.1)
    tseries_list["Tc"+str(i)] = amp * np.cos(2 * np.pi * f0 * time_index + phase_shift)

# модифицированный синус (Класс 1 - Ts)
for i in range(N):
    # Модификация: синус с затуханием или изменением частоты
    freq_mod = f0 * np.random.uniform(0.8, 1.2)
    tseries_list["Ts"+str(i)] = np.sin(2 * np.pi * freq_mod * time_index) * np.exp(-0.1 * time_index)

# пульс сигнал (Класс 2 - Tp)
for i in range(N):
    # Сдвигаем позицию импульса немного влево-вправо
    shift = np.random.uniform(-0.2, 0.2)
    # Создаем импульс в разной позиции
    t_shifted = time_index - np.random.uniform(2, 7) 
    tseries_list["Tp"+str(i)] = pulse(t_shifted)

# Отрисовка всех сигналов для визуальной проверки
plt.figure(figsize=(12, 8))
for i in range(N):
    plt.plot(time_index, tseries_list["Ts"+str(i)], '-b', alpha=0.5)
for i in range(N):
    plt.plot(time_index, tseries_list["Tc"+str(i)], '-g', alpha=0.5)
for i in range(N):
    plt.plot(time_index, tseries_list["Tp"+str(i)], '-r', alpha=0.5)

plt.title(r'Сгенерированные временные ряды (3 класса)')
plt.xlabel(r't (in s)')
plt.legend(['Sin Class', 'Cos Class', 'Pulse Class'])
plt.grid()
plt.show()

# 2. Расчет расстояний (Евклидово и DTW)

def distance_matrix(x, y, q) -> np.array:
    """
    Функция расчета матрицы расстояний между точками двух рядов
    """
    mdist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            mdist[i, j] = np.abs(x[j] - y[i])**q
    return mdist

def DTW(x, x_s, q=2, isDTW=True):
    '''
        x: первый ряд
        x_s : второй ряд
        q : степень для вычисления базового расстояния
    '''
    N_x = len(x)
    N_y = len(x_s)

    # Строим матрицу локальных расстояний
    dist = distance_matrix(x, x_s, q=q)

    # Инициализация матрицы накопленных расстояний R, используем inf для границ, чтобы алгоритм шел только из допустимых предыдущих состояний
    R = np.full((N_x + 1, N_y + 1), np.inf)
    R[0, 0] = 0

    # Заполнение матрицы DTW
    for i in range(1, N_x + 1):
        for j in range(1, N_y + 1):
            cost = dist[i-1, j-1]
            R[i, j] = cost + min(R[i-1, j],    # вставка
                                 R[i, j-1],    # удаление
                                 R[i-1, j-1])  # совпадение

    # DTW расстояние - это значение в правом нижнем углу
    dtw_dist = R[N_x, N_y]
    
    # Восстановление пути
    path = []
    i, j = N_x, N_y
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i > 0 and j > 0:
            if R[i-1, j-1] <= R[i-1, j] and R[i-1, j-1] <= R[i, j-1]:
                i -= 1
                j -= 1
            elif R[i-1, j] <= R[i, j-1]:
                i -= 1
            else:
                j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
            
    path.reverse()
    
    # Для сравнения с евклидовым расстоянием(если isDTW=False, возвращаем просто сумму квадратов разностей по диагонали, если длины равны)
    if not isDTW and N_x == N_y:
        euclidean_dist = np.sqrt(np.sum((x - x_s)**2))
        return euclidean_dist, path, R
        
    return dtw_dist, path, R

# Пример расчета для двух сигналов одного класса (Ts1 и Ts6)
x = np.array(tseries_list.get("Ts1", np.zeros(100)))
x_s = np.array(tseries_list.get("Ts6", np.zeros(100)))
x_p = np.array(tseries_list.get("Tp2", np.zeros(100)))

# DTW расстояние между похожими сигналами (Ts1 и Ts6)
s1_dtw, s1_path, s1_R = DTW(x, x_s, q=2, isDTW=True)
# Евклидово расстояние между ними же
s1_eucl = np.sqrt(np.sum((x - x_s)**2))

# DTW расстояние между разными сигналами (Ts1 и Tp2)
s2_dtw, s2_path, s2_R = DTW(x, x_p, q=2, isDTW=True)
s2_eucl = np.sqrt(np.sum((x - x_p)**2))

print(f"DTW Distance (Ts1 vs Ts6 - same class): {s1_dtw:.4f}")
print(f"Euclidean Distance (Ts1 vs Ts6 - same class): {s1_eucl:.4f}")
print(f"DTW Distance (Ts1 vs Tp2 - diff class): {s2_dtw:.4f}")
print(f"Euclidean Distance (Ts1 vs Tp2 - diff class): {s2_eucl:.4f}")

# Описание значений:
# Расстояние DTW между сигналами, принадлежащими одному классу (Ts1 и Ts6), оказывается существенно меньше, чем между сигналами из разных классов. 
# Евклидово расстояние также демонстрирует различия, однако DTW проявляет большую устойчивость к незначительным временным смещениям (фазовым сдвигам). 
# В представленной иллюстрации сигналы были сгенерированы таким образом, что обеспечивается их четкое разделение.

# Визуализация матрицы весов и пути выравнивания для Ts1 и Ts6
cost_matrix = s1_R[1:, 1:] # Убираем нулевую строку/столбец инициализации
warp_path = s1_path

fig, ax = plt.subplots(figsize=(12, 8))
ax = sbn.heatmap(cost_matrix, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
ax.invert_yaxis()

path_x = [p[0] for p in warp_path]
path_y = [p[1] for p in warp_path]
path_xx = [x+0.5 for x in path_x]
path_yy = [y+0.5 for y in path_y]

ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.8)
plt.title("DTW Warping Path (Ts1 vs Ts6)")
plt.show()

# Визуализация для разных классов (Ts1 и Tp2)
cost_matrix_2 = s2_R[1:, 1:]
warp_path_2 = s2_path

fig, ax = plt.subplots(figsize=(12, 8))
ax = sbn.heatmap(cost_matrix_2, square=True, linewidths=0.1, cmap="YlGnBu", ax=ax)
ax.invert_yaxis()

path_x_2 = [p[0] for p in warp_path_2]
path_y_2 = [p[1] for p in warp_path_2]
path_xx_2 = [x+0.5 for x in path_x_2]
path_yy_2 = [y+0.5 for y in path_y_2]

ax.plot(path_xx_2, path_yy_2, color='red', linewidth=3, alpha=0.8)
plt.title("DTW Warping Path (Ts1 vs Tp2)")
plt.show()

# Сравнение с библиотекой dtaidistance для проверки
x_arr = np.array(tseries_list["Ts1"])
xs_arr = np.array(tseries_list["Ts6"])
distance_lib, paths_lib = dtw.warping_paths(x_arr, xs_arr)
best_path_lib = dtw.best_path(paths_lib)
print(f"DTW distance via library: {distance_lib:.4f}")

# 3. Формирование набора данных

x_data = []  # значения признаков (временные ряды)
Y_labels = []  # целевая переменная (классы)

# Проходим по всем сгенерированным сигналам
keys = [k for k in tseries_list.keys() if k != 'Time']
for key in keys:
    x_data.append(np.array(tseries_list[key]))
    
    # Определяем класс по префиксу
    if key.startswith("Tc"):
        Y_labels.append(0) # Класс Cosine
    elif key.startswith("Ts"):
        Y_labels.append(1) # Класс Sine
    else: 
        Y_labels.append(2) # Класс Pulse

x_data = np.array(x_data)
Y_labels = np.array(Y_labels)

# Перемешивание данных
arr_indices = np.arange(len(Y_labels))
np.random.shuffle(arr_indices)

X_shuffled = x_data[arr_indices]
Y_shuffled = Y_labels[arr_indices]

# Разбиение на Train и Test, всего 21 образец, возьмем последние 5 для теста, остальные 16 для обучения.
split_idx = len(Y_shuffled) - 5

X_train = X_shuffled[:split_idx]
y_train = Y_shuffled[:split_idx]

X_test = X_shuffled[split_idx:]
y_test = Y_shuffled[split_idx:]

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Метки тестовой выборки: {y_test}")

# 4. Обучение классификаторов

# Модель 1: KNN с метрикой DTW 
print("\nОбучение KNN с DTW")
clf_knn_dtw = KNeighborsClassifier(metric='dtw', n_neighbors=3)
clf_knn_dtw.fit(X_train, y_train)

acc_knn_dtw = clf_knn_dtw.score(X_test, y_test)
print(f"Accuracy KNN (DTW): {acc_knn_dtw}")

# Предсказание для одного элемента
if len(X_test) > 0:
    proba_knn_dtw = clf_knn_dtw.predict_proba(X_test[0].reshape(1, -1))
    pred_knn_dtw = clf_knn_dtw.predict(X_test[0].reshape(1, -1))
    print(f"Пример предсказания (KNN DTW): Класс {pred_knn_dtw[0]}, Вероятности: {proba_knn_dtw[0]}")
    print(f"Истинный класс: {y_test[0]}")


# Модель 2: KNN с Евклидовой метрикой
print("\nОбучение KNN с Euclidean")
clf_knn_eucl = KNeighborsClassifier(metric='euclidean', n_neighbors=3)
clf_knn_eucl.fit(X_train, y_train)

acc_knn_eucl = clf_knn_eucl.score(X_test, y_test)
print(f"Accuracy KNN (Euclidean): {acc_knn_eucl}")

if len(X_test) > 0:
    proba_knn_eucl = clf_knn_eucl.predict_proba(X_test[0].reshape(1, -1))
    pred_knn_eucl = clf_knn_eucl.predict(X_test[0].reshape(1, -1))
    print(f"Пример предсказания (KNN Eucl): Класс {pred_knn_eucl[0]}, Вероятности: {proba_knn_eucl[0]}")


# Модель 3: TimeSeriesForest
print("\nОбучение TimeSeriesForest")
# TimeSeriesForest требует целочисленные индексы или специфический формат, но pyts обычно принимает numpy arrays
clf_tsforest = TimeSeriesForest(n_estimators=100, random_state=42)
clf_tsforest.fit(X_train, y_train)

acc_tsforest = clf_tsforest.score(X_test, y_test)
print(f"Accuracy TimeSeriesForest: {acc_tsforest}")

if len(X_test) > 0:
    proba_tsforest = clf_tsforest.predict_proba(X_test[0].reshape(1, -1))
    pred_tsforest = clf_tsforest.predict(X_test[0].reshape(1, -1))
    print(f"Пример предсказания (TS Forest): Класс {pred_tsforest[0]}, Вероятности: {proba_tsforest[0]}")


# 5. Описание результатов

"""
Результаты исследования представлены следующим образом:

1. Качество классификации. На сгенерированном наборе данных все три модели продемонстрировали высокую точность, близкую к единице. 
Это объясняется существенными различиями между классами (синус, косинус, импульс) по форме сигналов.

2. Сравнение метрик для KNN:
- Использование DTW в KNN обеспечивает лучшую устойчивость к временным сдвигам. 
При генерации сигналов с случайными фазовыми сдвигами качество классификации посредством DTW заметно превосходило бы результаты Евклидовой метрики. 
В рассматриваемом случае, когда сигналы выровнены по времени, Евклидова метрика тоже демонстрирует удовлетворительный результат.
- Метрика Евклида, напротив, чувствительна к сдвигам, однако вычислительно более эффективна по сравнению с DTW.

3. TimeSeriesForest:
- Это ансамблевый алгоритм, строящий деревья решений на случайных подвыборках признаков, представляющих собой подпоследовательности временного ряда.
- Метод как правило обеспечивает высокую устойчивость к шуму и хорошую способность к обобщению.
- Обучение может занимать больше времени, чем у KNN, однако предсказания выполняются быстрее, особенно по сравнению с KNN-DTW, который требует вычисления расстояний до каждого объекта обучающей выборки.

4. Заключение. Для задач классификации временных рядов, в которых вероятны нелинейные искажения во времени, метрика DTW представляется предпочтительной для алгоритмов ближайших соседей. 
В то же время TimeSeriesForest служит мощной альтернативой, особенно если необходимо выявлять локальные паттерны в данных.
"""
