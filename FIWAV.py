import numpy as np
import scipy.io.wavfile as wav

def generar_filtro_inverso(input_file, output_file, fs=44100):
    # Cargar el archivo .wav del sine sweep
    fs_sine_sweep, sine_sweep = wav.read(input_file)

    duracion = len(sine_sweep) / fs_sine_sweep  # Duración del sine sweep

    t_swipe_arange = np.arange(0, duracion, 1/fs)  # Arreglo de tiempo
    R = np.log(1)  # El logaritmo de 1 es 0 ya que la frecuencia inicial y final son iguales
    K = 2*np.pi
    L = duracion
    w = K*np.exp(t_swipe_arange/L)
    m = 1/w

    # Calcula el filtro inverso k(t)
    k_t = m * sine_sweep[::-1]  # Inversion temporal de x(t)

    # Normaliza el Filtro Inverso
    k_t /= np.max(np.abs(k_t))

    # Guarda el filtro inverso k(t) como archivo de audio .wav
    wav.write(output_file, fs, k_t.astype(np.float32))

# Uso de la función
input_file = 'toma_n1_a-03.wav'  # Nombre del archivo .wav del sine sweep
output_file = 'filtro_inversoDR.wav'  # Nombre del archivo de salida del filtro inverso

generar_filtro_inverso(input_file, output_file)