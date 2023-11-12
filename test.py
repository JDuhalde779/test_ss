import os
import numpy as np
import wave
from pydub import AudioSegment
import scipy.signal as signal
from scipy.signal import find_peaks
from pydub.playback import play
import soundfile as sf
import scipy.io.wavfile as wav



def iec61260_filtros(audio_signal, center_frequency, sample_rate=44100):
    """
   Aplica filtros acústicos según la norma IEC 61260 a una señal de audio.

   Esta función toma una señal de audio, la frecuencia central deseada, y opcionalmente la frecuencia de muestreo,
   y aplica un filtro acústico de octava o tercio de octava según la norma IEC 61260.
   La función guarda la señal filtrada en un archivo WAV individual con un nombre apropiado para la frecuencia central.

   Parámetros:
   audio_signal (array): La señal de audio de entrada.
   center_frequency (float): La frecuencia central a la cual se aplicará el filtro.
   sample_rate (int, opcional): La frecuencia de muestreo de la señal de audio. Valor predeterminado: 44100.

   La función guarda la señal filtrada en un archivo WAV individual con un nombre apropiado para la frecuencia central.
   
   Args:
       audio_signal (array): La señal de audio de entrada.
       center_frequency (float): La frecuencia central a la cual se aplicará el filtro.
       sample_rate (int, opcional): La frecuencia de muestreo de la señal de audio.

   Returns:
       None
   """
    # Lista de frecuencias centrales según la norma IEC61260 para octavas y tercios de octava
    center_frequencies = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    frecuencias_centrales_tercio = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]

    if center_frequency in center_frequencies:
        G = 1.0 / 2.0  # Octava - G = 1.0/2.0 / 1/3 de Octava - G=1.0/6.0
        factor = np.power(2, G)
        center_frequency_hz = center_frequency

        lower_cutoff_frequency_hz = center_frequency_hz / factor
        upper_cutoff_frequency_hz = center_frequency_hz * factor

        # Para aplicar el filtro de manera más óptima
        sos = signal.iirfilter(4, [lower_cutoff_frequency_hz, upper_cutoff_frequency_hz],
                               rs=60, btype='band', analog=False,
                               ftype='butter', fs=sample_rate, output='sos')
        filtered_signal = signal.sosfilt(sos, audio_signal)

        # Guarda la señal filtrada en un archivo individual
        sf.write(f"señal_filtrada_{center_frequency}.wav", filtered_signal, sample_rate)
    
    if center_frequency in frecuencias_centrales_tercio:
        G = 1.0 / 6.0  # Octava - G = 1.0/2.0 / 1/3 de Octava - G=1.0/6.0
        factor = np.power(2, G)
        center_frequency_hz = center_frequency

        lower_cutoff_frequency_hz = center_frequency_hz / factor
        upper_cutoff_frequency_hz = center_frequency_hz * factor

        # Para aplicar el filtro de manera más óptima
        sos = signal.iirfilter(4, [lower_cutoff_frequency_hz, upper_cutoff_frequency_hz],
                               rs=60, btype='band', analog=False,
                               ftype='butter', fs=sample_rate, output='sos')
        filtered_signal = signal.sosfilt(sos, audio_signal)

        # Guarda la señal filtrada en un archivo individual
        sf.write(f"señal_filtrada_tercio_{center_frequency}.wav", filtered_signal, sample_rate)
    
    else:
        print("Se ha ingresado un valor de frecuencia inválido")

# Llamar a la función con alguna RI generada anteriormente.
audio_signal, sample_rate = sf.read("respuesta_al_impulsoObtenida.wav")
iec61260_filtros(audio_signal,1000, sample_rate=44100)


def regresion_lineal(x, y):
    # Función para realizar la regresión lineal
    X = np.vstack((np.ones(len(x)), x)).T
    XT = np.transpose(X)
    XTX = np.dot(XT, X)
    XTX_inv = np.linalg.inv(XTX)
    XTY = np.dot(XT, y)
    coeficientes = np.dot(XTX_inv, XTY)
    return coeficientes

# Cargar la respuesta al impulso desde el archivo .wav
respuesta_al_impulso, tasa_muestreo = sf.read("salida_filtrada.wav")

# Crear un vector de tiempo (puedes ajustar esto según la duración de tu respuesta al impulso)
tiempo = np.arange(0, len(respuesta_al_impulso)/tasa_muestreo, 1/tasa_muestreo)

# Aplicar regresión lineal
coeficientes = regresion_lineal(tiempo, respuesta_al_impulso)

# Coeficientes resultantes: coeficientes[0] es b (ordenada al origen) y coeficientes[1] es a (pendiente)
a, b = coeficientes[1], coeficientes[0]

print(f"La recta de regresión es: y = {a}x + {b}")   


def calcular_edt_con_regresion(ruta_archivo, umbral_dB=10, ventana_suavizado=100):
    # Función para calcular el EDT utilizando la regresión lineal para determinar el tiempo donde la respuesta cae 10 dB desde el pico
    tasa_muestreo, señal = wav.read(ruta_archivo)


    # Encontrar el índice del pico en la región de interés
    pico_indice = np.argmax(señal)

    # Encontrar el punto donde la respuesta al impulso suavizada decae 10 dB desde el pico
    umbral_amplitud = np.max(señal) - umbral_dB
    indices_descenso, _ = find_peaks(-señal[pico_indice:], height=umbral_amplitud)

    # Calcular el tiempo de EDT utilizando la regresión lineal
    if indices_descenso.size > 0:
        tiempo_edt = pico_indice + indices_descenso[0]
        tiempo_edt /= tasa_muestreo

        # Calcular la regresión lineal para la parte posterior al pico
        x_post_pico = np.arange(pico_indice, len(señal)) / tasa_muestreo
        y_post_pico = señal[pico_indice:]

        # Calcular el tiempo de decaimiento usando la regresión lineal
        tiempo_decaimiento_regresion = -umbral_dB / coeficientes[1]
        tiempo_decaimiento_regresion += tiempo_edt
        print(f"Tiempo de decaimiento (regresión): {tiempo_decaimiento_regresion} segundos")
    else:
        print("No se pudo calcular el tiempo de decaimiento utilizando la regresión lineal.")

# Ejemplo de uso
output_file = "salida_filtrada.wav"  # Reemplazar con la ruta correcta de tu archivo WAV
calcular_edt_con_regresion(output_file)




