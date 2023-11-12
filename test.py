import os
import numpy as np
import wave
from pydub import AudioSegment
import scipy.signal as signal
from scipy.signal import find_peaks
from pydub.playback import play
import soundfile as sf
import scipy.io.wavfile as wav



def calcular_edt(ruta_archivo):
    """
    Calcula el Early Decay Time (EDT) de un archivo WAV.

    Parámetros:
    - ruta_archivo: Ruta del archivo WAV.

    Devuelve el EDT en segundos.
    """
    tasa_muestreo = 44100

    # Encontrar el pico de la respuesta al impulso
    pico_indice = np.argmax(ruta_archivo)

    # Encontrar el punto donde la respuesta al impulso decae 10 dB desde el pico
    umbral_dB = -10
    indices_descenso, _ = find_peaks(-ruta_archivo[pico_indice:], height=umbral_dB)

    # Calcular el EDT en segundos
    edt_indice = pico_indice + indices_descenso[0]
    tiempo_edt = edt_indice / tasa_muestreo

    return tiempo_edt

# Ejemplo de uso
ruta_archivo_wav = "ejemplo.wav"  # Reemplaza con la ruta de tu archivo WAV
edt_resultado = calcular_edt(ruta_archivo_wav)
print(f"EDT: {edt_resultado} segundos")