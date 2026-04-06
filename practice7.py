
# Задание №1: Теоретические вопросы

"""
1. Сравнение SAX и SFA:
   SAX (Symbolic Aggregate approXimation) базируется на методе PAA, который усредняет данные во временной области, после чего выполняется квантование на основе нормального распределения. 
   В отличие от него, SFA (Symbolic Fourier Approximation) применяет дискретное преобразование Фурье для извлечения признаков на низких частотах, с последующим квантованием методом MFBF. 
   При обработке зашумленных данных SFA показывает лучшие результаты, поскольку анализ фокусируется на первых коэффициентах Фурье, таким образом игнорируя высокочастотный шум. 
   SAX же более восприимчив к локальному шуму из-за усреднения по временной оси.

2. Методы BOSS/WEASEL и использование TF-IDF:
   Метод BOSS преобразует временные ряды в представление с помощью мешка слов, формируя гистограммы частот появления различных паттернов. 
   WEASEL расширяет эту концепцию, применяя TF-IDF (Term Frequency-Inverse Document Frequency) для снижения вклада слишком частотных паттернов, присутствующих во всех классах и не обладающих дискриминационной способностью. 
   Это повышает значимость уникальных паттернов, характерных для конкретных классов.

3. Подходы Shapelet и ROCKET:
   Shapelet представляет собой подпоследовательность временного ряда, несущую максимальную информацию для различения классов, характеризующуюся наибольшим расстоянием между классами. 
   Однако классический алгоритм перебора всех возможных подпоследовательностей чрезвычайно затратен по времени. 
   В качестве альтернативы ROCKET применяет множество случайных сверточных фильтров, что приближенно соответствует поиску паттернов или шейплетов, но позволяет выполнять вычисления значительно быстрее за счет случайной инициализации фильтров и отсутствия фазы обучения их весов — обучение сводится к линейной классификации на выходных признаках.

4. Сравнение catch22 и глубокого обучения:
   Метод catch22 извлекает 22 статистических признака, включающих, например, среднее, дисперсию и энтропию. 
   Его преимущество проявляется в задачах с ограниченным объемом данных, где глубокие сверточные нейросети (CNN) требуют значительных ресурсов, включая большие дата-сеты и тщательную настройку гиперпараметров. 
   catch22 обладает высокой скоростью работы, обеспечивает интерпретируемость результатов и часто достигает сопоставимой с CNN эффективности на малых выборках без необходимости использования GPU.

5. Особенности 1D-CNN и мультиветвевой архитектуры (Inception):
   Одномерные сверточные сети способны автоматически выявлять иерархические признаки: от локальных паттернов на нижних слоях до глобальных абстракций на верхних уровнях. 
   Мультиветвевые модели, подобные InceptionTime, параллельно применяют сверточные ядра различных размеров, что позволяет сети одновременно фиксировать структуры разной длительности и масштаба. 
   Такой подход обходится без необходимости предварительного задания оптимального масштаба для анализируемых данных.
"""

# Задание №2: Практическая реализация

# Загрузка данных
from sktime.datasets import load_arrow_head
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import time

# Мы используем датасет ArrowHead
X, y = load_arrow_head(return_type="numpy3d")
# X имеет форму (Кол-во образцов, кол-во каналов=1, длина ряда)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Задание 2.1: ROCKET
print("\nЗадание 2.1: ROCKET-")
from sktime.classification.kernel_based import RocketClassifier

# 1. Инициализация и обучение
start_time = time.time()
rocket_clf = RocketClassifier(num_kernels=1000) # Стандартное количество ядер
rocket_clf.fit(X_train, y_train)
train_time_rocket = time.time() - start_time

# 2. Предсказание и оценка
start_time = time.time()
y_pred_rocket = rocket_clf.predict(X_test)
pred_time_rocket = time.time() - start_time

acc_rocket = accuracy_score(y_test, y_pred_rocket)
print(f"ROCKET Accuracy: {acc_rocket:.4f}")
print(f"ROCKET Train Time: {train_time_rocket:.4f} sec")
print(f"ROCKET Predict Time: {pred_time_rocket:.4f} sec")


# Задание 2.2: Dictionary-based (cBOSS)
print("\nЗадание 2.2: ContractableBOSS")
from sktime.classification.dictionary_based import ContractableBOSS

# 1. Инициализация и обучение, cBOSS может работать долго, поэтому ограничим время или количество параметров для примера
start_time = time.time()
cboss_clf = ContractableBOSS(max_win_len_prop=0.5, n_parameter_samples=4) # Уменьшенные параметры для скорости
cboss_clf.fit(X_train, y_train)
train_time_cboss = time.time() - start_time

# 2. Предсказание и оценка
start_time = time.time()
y_pred_cboss = cboss_clf.predict(X_test)
pred_time_cboss = time.time() - start_time

acc_cboss = accuracy_score(y_test, y_pred_cboss)
print(f"cBOSS Accuracy: {acc_cboss:.4f}")
print(f"cBOSS Train Time: {train_time_cboss:.4f} sec")
print(f"cBOSS Predict Time: {pred_time_cboss:.4f} sec")


# Задание 2.3: 1D-CNN (PyTorch)
print("\nЗадание 2.3: 1D-CNN (PyTorch)")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Подготовка данных для PyTorch
# sktime выдает данные в формате (N, C, L). PyTorch Conv1d ожидает (N, C, L).
# Нам нужно закодировать метки классов в числа (0, 1, 2...)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Преобразуем в тензоры
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_enc, dtype=torch.long)
y_test_torch = torch.tensor(y_test_enc, dtype=torch.long)

# Создаем DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_torch, y_test_torch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Определение модели
class Simple1DCNN(nn.Module):
    def __init__(self, num_classes, input_length):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        # Динамический расчет размера после пулинга
        # Length after conv1: L - k + 1
        # Length after pool1: floor((L - k + 1) / 2)
        # Length after conv2: L2 - k + 1
        # Length after pool2: floor((L2 - k + 1) / 2)
        
        l1 = input_length - 3 + 1
        l2 = l1 // 2
        l3 = l2 - 3 + 1
        l4 = l3 // 2
        
        self.fc_input_size = 64 * l4
        
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Параметры
num_classes = len(np.unique(y))
input_length = X_train.shape[2] # Длина временного ряда

model = Simple1DCNN(num_classes=num_classes, input_length=input_length)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение
num_epochs = 20
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

train_time_cnn = time.time() - start_time

# Оценка
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Декодируем предсказания обратно в оригинальные лейблы
y_pred_cnn = le.inverse_transform(all_preds)
acc_cnn = accuracy_score(y_test, y_pred_cnn)

print(f"1D-CNN Accuracy: {acc_cnn:.4f}")
print(f"1D-CNN Train Time: {train_time_cnn:.4f} sec")

# Сравнение результатов
print("\nИтоговое сравнение")
print(f"{'Метод':<15} | {'Accuracy':<10} | {'Train Time (s)':<15}")
print("-" * 45)
print(f"{'ROCKET':<15} | {acc_rocket:<10.4f} | {train_time_rocket:<15.4f}")
print(f"{'cBOSS':<15} | {acc_cboss:<10.4f} | {train_time_cboss:<15.4f}")
print(f"{'1D-CNN':<15} | {acc_cnn:<10.4f} | {train_time_cnn:<15.4f}")
