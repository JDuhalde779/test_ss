import os
import numpy as np
import soundfile as sf
import wave
from scipy.signal import find_peaks
from pydub.playback import play
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from pydub import AudioSegment
import pandas as pd
from scipy.signal import welch
import sounddevice as sd
import time
import scipy.io.wavfile as wav
import scipy.signal as signal



def plot_dominio_temporal(señal, fs=44100, inicio=None, duracion=None, umbral_amplitud=None):
    """
    Muestra el dominio temporal de la señal con un umbral de amplitud.

    Parámetros
    ----------
    señal : str o NumPy array
        Si es una cadena (str), se asume que es una ruta al archivo de audio WAV.
        Si es un NumPy array, se asume que es la señal directa.
    fs : int
        Frecuencia de muestreo en Hz de la señal.
    inicio : float, opcional
        Tiempo de inicio para la ventana en segundos.
    duracion : float, opcional
        Duración de la ventana en segundos.
    umbral_amplitud : float, opcional
        Umbral de amplitud para mostrar valores en el gráfico.

    Retorna
    -------
    None
    """
    if isinstance(señal, str):  # Si es una cadena, se asume que es un archivo WAV
        tasa_muestreo, audio_data = wav.read(señal)
        fs = tasa_muestreo
    else:  # Si es un NumPy array, se asume que es la señal directa
        audio_data = señal

    # Calcula los valores de tiempo
    tiempo = np.arange(len(audio_data)) / fs

    # Establece el índice de inicio y final
    if inicio is None:
        inicio = 0
    if duracion is None:
        duracion = tiempo[-1]

    # Encuentra los índices correspondientes al inicio y final de la ventana
    inicio_idx = int(inicio * fs)
    fin_idx = int((inicio + duracion) * fs)

    # Asegura que los índices estén dentro de los límites de la señal
    inicio_idx = max(0, inicio_idx)
    fin_idx = min(len(audio_data), fin_idx)

    # Aplicar umbral de amplitud si se proporciona
    if umbral_amplitud is not None:
        audio_data[audio_data < umbral_amplitud] = umbral_amplitud

    # Crea una nueva figura y plotea la señal en la ventana especificada
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo[inicio_idx:fin_idx], audio_data[inicio_idx:fin_idx])
    plt.title('Dominio Temporal de la Señal')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()



def acortar_wav(input_path, output_path, duracion_deseada):
    # Cargar el archivo WAV
    audio = AudioSegment.from_wav(input_path)

    # Acortar la duración según lo deseado
    audio_recortado = audio[:duracion_deseada * 1000]  # Duración en milisegundos

    # Guardar el nuevo archivo WAV
    audio_recortado.export(output_path, format="wav")

# Especifica la ruta del archivo de entrada y salida, y la nueva duración deseada en segundos
archivo_entrada = "respuesta_al_impulsoObtenida.wav"
archivo_salida = "impulso_recortado.wav"
duracion_deseada = 5  # Por ejemplo, 10 segundos    


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
audio_signal, sample_rate = sf.read("respuesta_al_impulsoDESCARGADOS.wav")
frecuencias_centrales = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
for i in frecuencias_centrales:
    iec61260_filtros(audio_signal,i, sample_rate=44100)


def stereo_a_mono_wav(archivo_entrada):

    # Verifica si el archivo de entrada es un archivo WAV
    if not archivo_entrada.lower().endswith(".wav"):
        print("El archivo de entrada debe ser un archivo WAV.")
        return

    # Construye el nombre del archivo de salida con "_mono" agregado
    output_file = os.path.splitext(archivo_entrada)[0] + "_mono.wav"

    # Carga el archivo de audio estéreo en formato WAV
    audio = AudioSegment.from_wav(archivo_entrada)

    # Convierte el audio a mono
    audio = audio.set_channels(1)

    # Guarda el audio mono en el archivo de salida en formato WAV
    audio.export(output_file, format="wav")
    print(f"Archivo de salida guardado como {output_file}")
    return output_file







