import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
from scipy.fft import fft, ifft



def generar_filtro_inverso(input_file, output_file, fs=44100):
    # Cargar el archivo .wav del sine sweep
    fs_sine_sweep, sine_sweep = wav.read(input_file)

    duracion = len(sine_sweep) / fs_sine_sweep  # Duración del sine sweep

    t_swipe_arange = np.arange(0, duracion*fs)/fs  # Arreglo de muestreos
    R = np.log(20/20000)  
    K = duracion*2*np.pi*20/R
    L = duracion/R
    w = K*np.exp(t_swipe_arange/L)
    m = 20/w

    # Calcula el filtro inverso k(t)
    k_t = m * sine_sweep[::-1]  # Inversion temporal de x(t)

    # Normaliza el Filtro Inverso
    k_t /= np.max(np.abs(k_t))

    # Guarda el filtro inverso k(t) como archivo de audio .wav
    wav.write(output_file, fs, k_t.astype(np.float32))

# Uso de la función
input_file = 'Toma_n1_d-03.wav'  # Nombre del archivo .wav del sine sweep
output_file = 'filtro_inversoDR.wav'  # Nombre del archivo de salida del filtro inverso

#generar_filtro_inverso(input_file, output_file)

def generar_respuesta_al_impulso(T60_lista, frecuencias_lista, duracion, archivo_salida):
    # Número de muestras
    fs = 44100  # Frecuencia de muestreo (puedes ajustarla)
    n_muestras = int(fs * duracion)

    # Crear el arreglo de tiempo
    t = np.arange(0, n_muestras) / fs

    # Inicializar la respuesta al impulso
    respuesta_impulso = np.zeros(n_muestras)

    # Generar la respuesta al impulso para cada frecuencia y T60 y sumarlas
    for T60, frecuencia in zip(T60_lista, frecuencias_lista):
        # Cálculo de τ
        tau = -np.log(10**(-3)) / (T60)

        # Generar la respuesta al impulso para la frecuencia actual
        respuesta_frecuencia = np.exp(-tau * t) * np.cos(2 * np.pi * frecuencia * t)

        # Sumar la respuesta de esta frecuencia a la respuesta global
        respuesta_impulso += respuesta_frecuencia

    # Normalizar la respuesta al impulso
    respuesta_impulso /= np.max(np.abs(respuesta_impulso))

    # Guardar la respuesta al impulso como archivo de audio .wav
    wav.write(archivo_salida, fs, respuesta_impulso.astype(np.float32))

# Ejemplo de uso de la función
T60_lista = [2.59, 2.61, 2.47]  # Lista de T60 para cada frecuencia
frecuencias_lista = [500, 1000, 2000]  # Lista de frecuencias centrales
duracion = max(T60_lista) + 2  # Duración en segundos
nombre_archivo = 'respuesta_al_impulso.wav'  # Nombre del archivo de salida

#generar_respuesta_al_impulso(T60_lista, frecuencias_lista, duracion, nombre_archivo)

def respuesta_al_impulso(sine_sweep_wav, filtro_inverso_wav, salida_wav):
    # Cargar los archivos .wav
    sine_sweep, fs_sine_sweep = sf.read(sine_sweep_wav)
    filtro_inverso, fs_filtro_inverso = sf.read(filtro_inverso_wav)

    # Asegurarse de que tengan la misma frecuencia de muestreo
    if fs_sine_sweep != fs_filtro_inverso:
        raise ValueError("Las señales deben tener la misma frecuencia de muestreo.")

    # Calcular la transformada de Fourier de ambas señales
    fft_sine_sweep = fft(sine_sweep)
    fft_filtro_inverso = fft(filtro_inverso)

    # Multiplicar las transformadas en el dominio de la frecuencia
    respuesta_frecuencial = fft_sine_sweep * fft_filtro_inverso

    # Aplicar la antitransformada para obtener la respuesta al impulso en el dominio del tiempo
    respuesta_impulso = ifft(respuesta_frecuencial).real

    # Normalizar la respuesta al impulso
    respuesta_impulso /= np.max(np.abs(respuesta_impulso))

    # Guardar la respuesta al impulso como archivo de audio .wav
    sf.write(salida_wav, respuesta_impulso, fs_sine_sweep)

# Ejemplo de uso de la función
sine_sweep_wav = 'Toma_n1_d-03.wav'  # Archivo .wav del sine sweep logarítmico
filtro_inverso_wav = 'filtro_inversoDR.wav'  # Archivo .wav del filtro inverso
salida_wav = 'respuesta_al_impulsoObtenida.wav'  # Nombre del archivo de salida de la respuesta al impulso

respuesta_al_impulso(sine_sweep_wav, filtro_inverso_wav, salida_wav)



