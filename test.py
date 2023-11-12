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










