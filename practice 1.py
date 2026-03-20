# глобальные переменные, которые нам нужны для генерации всех шумов
duration = 5  # Продолжительность в секундах
sample_rate = 4100  # Частота дискретизации в Гц

# считаем, сколько элементов будем генерировать
num_samples = int(duration * sample_rate)

# генерируем белый шум
white_noise_g = np.random.normal(loc=0, scale=1, size=num_samples)

# преобразование Фурье
fft_white = np.fft.rfft(white_noise_g)

# частотная шкала
frequencies = np.fft.rfftfreq(num_samples, d=1/sample_rate)

# деление амплитуд на квадратный корень из частоты
frequencies[0] = 1  # заменяем 0 на 1, чтобы избежать деления на ноль
fft_pink = fft_white / np.sqrt(frequencies)

# обратное преобразование Фурье
pink_noise = np.fft.irfft(fft_pink, n=num_samples)

# нормализация сигнала
pink_noise = pink_noise / np.std(pink_noise)

print(pink_noise)

# отрисовываем
time_axis = np.linspace(0, duration, len(pink_noise))
plt.figure(figsize=(20, 6))
plt.plot(time_axis, pink_noise)
plt.title('Розовый шум')
plt.xlabel('Время (секунды)')
plt.ylabel('Амплитуда')
plt.show()

# слушаем, что получилось
sd.play(pink_noise, sample_rate)
sd.wait()
