import os
import numpy as np
import wave
from pydub import AudioSegment
import scipy.signal as signal
from scipy.signal import find_peaks
from pydub.playback import play
import soundfile as sf
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
from pre_procesamiento import plot_dominio_temporal
def convertir_audio_a_escala_logaritmica(señal_audio):
    """
    Convierte un archivo de audio en escala logarítmica y devuelve el resultado como un array.

    Parámetros:
    señal_audio (str): Ruta al archivo de audio de entrada (formato .wav).

    Retorna:
    numpy.ndarray: El array de la señal en escala logarítmica.
    """
    # Cargar el archivo de audio .wav
    tasa_muestreo, audio_data = wav.read(señal_audio)

    # Normalizar los valores de audio entre -1 y 1
    audio_data = audio_data.astype(np.float32) / 32767.0

    # Aplicar la conversión logarítmica
    audio_log = 20 * np.log10(np.abs(audio_data))

    return audio_log


def filtro_promedio_movil(input_file, output_file, L):
    # Leer el archivo WAV de entrada
    sample_rate, audio_data = wav.read(input_file)

    # Aplicar el filtro de promedio móvil
    filtered_signal = np.zeros(len(audio_data))
    for i in range(L, len(audio_data)):
        filtered_signal[i] = (1/L) * sum(audio_data[i-j] for j in range(L))

    # Guardar la señal filtrada en un archivo WAV de salida
    wav.write(output_file, sample_rate, filtered_signal.astype(np.int16))

# Ejemplo de uso
input_file = "respuesta_al_impulsoDESCARGADOS.wav"
output_file = "salida_filtrada.wav"
L = 10  # Número de muestras para el promedio móvil
filtro_promedio_movil(input_file, output_file, L)
señal_audio = output_file
audio_log = convertir_audio_a_escala_logaritmica(señal_audio)
señal_audio_2 = input_file
audio_log_2 = convertir_audio_a_escala_logaritmica(señal_audio_2)
plot_dominio_temporal(audio_log_2, fs=44100, inicio=0, duracion=1.75, umbral_amplitud= None)
plot_dominio_temporal(audio_log, fs=44100, inicio=0, duracion=1.75, umbral_amplitud= None)






