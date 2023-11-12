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
from Funciones_notebook_2 import cargar_archivos_de_audio
from Funciones_notebook_2 import stereo_a_mono_wav

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

    epsilon = 1e-10

    # Calcular el espectro en escala logarítmica (en decibelios)
    audio_log = 20 * np.log10(np.abs(audio_data) + epsilon)

    return audio_log


archivos_audio = cargar_archivos_de_audio("carpeta_de_audios")

# Imprime la lista de archivos
for i, archivo in enumerate(archivos_audio):
    print(f"{i}: {archivo}")

# Elije el índice del archivo que deseas cargar
indice_archivo = 0  # Cambia este valor al índice del archivo que deseas cargar

if 0 <= indice_archivo < len(archivos_audio):
    archivo_seleccionado = archivos_audio[indice_archivo]
    print(f"Cargando archivo: {archivo_seleccionado}")

    # Llama a la función con el archivo seleccionado
    # Luego puedes realizar operaciones con el archivo, por ejemplo, cargarlo y procesarlo
    # Cargar el archivo, procesarlo, etc.
else:
    print("El índice seleccionado está fuera de rango. Debe estar en el rango [0, {}].".format(len(archivos_audio) - 1))
# Ejemplo de uso


archivo_entrada = archivo_seleccionado
archivo_mono = stereo_a_mono_wav(archivo_entrada)

def filtro_promedio_movil(input_file, output_file, L):
    # Leer el archivo WAV de entrada
    sample_rate, audio_data = wav.read(input_file)
   
    # Aplicar el filtro de promedio móvil
    filtered_signal = np.zeros_like(audio_data, dtype=np.float64)

    for i in range(L, len(audio_data)):
        filtered_signal[i] = (1/L) * np.sum(audio_data[i-L+1:i+1])

    # Guardar la señal filtrada en un archivo WAV de salida
    wav.write(output_file, sample_rate, filtered_signal.astype(np.int16))


# Ejemplo de uso
input_file = "respuesta_al_impulsoObtenida.wav"
output_file = "salida_filtrada.wav"
L = 100 # Número de muestras para el promedio móvil
filtro_promedio_movil(input_file, output_file, L)
señal_audio = output_file
audio_log = convertir_audio_a_escala_logaritmica(señal_audio)
señal_audio_2 = input_file
audio_log_2 = convertir_audio_a_escala_logaritmica(señal_audio_2)
plot_dominio_temporal(audio_log_2, fs=44100, inicio=0, duracion=1.75, umbral_amplitud= -100)
plot_dominio_temporal(audio_log, fs=44100, inicio=0, duracion=1.75, umbral_amplitud= -100)


def calcular_edt_para_todas_las_frecuencias(ruta_archivo, frecuencias_tercios_octava):
    # Función para calcular el EDT para todas las frecuencias de tercios de octava de un archivo WAV
    tasa_muestreo, señal = wav.read(ruta_archivo)
    resultados_edt = []

    for frecuencia in frecuencias_tercios_octava:
        # Encontrar el índice de la frecuencia más cercana
        indice_frecuencia_cercana = int(frecuencia / (tasa_muestreo / len(señal)))

        pico_indice = np.argmax(señal[indice_frecuencia_cercana:])

        # Encontrar el punto donde la respuesta al impulso suavizada decae 10 dB desde el pico
        umbral_dB = -10
        indices_descenso, _ = find_peaks(-señal[pico_indice:], height=umbral_dB)

        # Calcular el EDT en segundos
        edt_indice = pico_indice + indices_descenso[0]
        tiempo_edt = edt_indice / tasa_muestreo

        resultados_edt.append(tiempo_edt)

    return resultados_edt

# Ejemplo de uso
frecuencias_tercios_octava = [125, 250, 500, 1000, 2000, 4000, 8000]  # Ejemplo de frecuencias de tercios de octava

resultados_edt = calcular_edt_para_todas_las_frecuencias(output_file, frecuencias_tercios_octava)
for frecuencia, edt_resultado in zip(frecuencias_tercios_octava, resultados_edt):
    print(f"EDT para {frecuencia} Hz: {edt_resultado} segundos")