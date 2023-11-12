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

#Constantes
fs = 44100

def cargar_archivos_de_audio(directorio):
    archivos_de_audio = []

    # Verifica si el directorio existe
    if not os.path.exists(directorio):
        print(f"El directorio '{directorio}' no existe.")
        return archivos_de_audio

    # Itera sobre los archivos en el directorio
    for archivo in os.listdir(directorio):
        if archivo.endswith(".wav"):
            archivos_de_audio.append(os.path.join(directorio, archivo))

    return archivos_de_audio

# Directorio que contiene los archivos de audio .wav
# Utiliza una ruta relativa desde el directorio donde se encuentra este script
directorio_audio = "carpeta_de_audios"

# Carga los archivos de audio
archivos_audio = cargar_archivos_de_audio(directorio_audio)

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

# Ejemplo de uso
archivo_entrada = archivo_seleccionado
stereo_a_mono_wav(archivo_entrada)


def generar_filtro_inverso(input_file, output_file, fs=44100):
    # Cargar el archivo .wav del sine sweep
    fs_sine_sweep, sine_sweep = wav.read(input_file)

    duracion = len(sine_sweep) / fs_sine_sweep  # Duración del sine sweep

    t_swipe_arange = np.arange(0, duracion*fs)/fs  # Arreglo de muestreos
    R = np.log(20000/20)  
    K = duracion*2*np.pi*20/R
    L = duracion/R
    w = (K/L)*np.exp(t_swipe_arange/L)
    m = 20/w

    # Calcula el filtro inverso k(t)
    k_t = m * sine_sweep[::-1]  # Inversion temporal de x(t)

    # Normaliza el Filtro Inverso
    k_t /= np.max(np.abs(k_t))

    # Guarda el filtro inverso k(t) como archivo de audio .wav
    wav.write(output_file, fs, k_t.astype(np.float32))

# Uso de la función
input_file = 'Toma_n1_c-03.wav'  # Nombre del archivo .wav del sine sweep
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


# Reproducir la respuesta al impulso generada
#play(AudioSegment.from_wav(nombre_archivo))


sine_sweep = "signal_recording.wav"
filtro_inv = "filtro_inverso.wav"

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
sine_sweep_wav = 'Toma_n1_c-03.wav'  # Archivo .wav del sine sweep logarítmico
filtro_inverso_wav = 'filtro_inversoDR.wav'  # Archivo .wav del filtro inverso
salida_wav = 'respuesta_al_impulsoDESCARGADOS.wav'  # Nombre del archivo de salida de la respuesta al impulso

#respuesta_al_impulso(sine_sweep_wav, filtro_inverso_wav, salida_wav)



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


señal_audio = "respuesta_al_impulsoObtenida.wav"
audio_log = convertir_audio_a_escala_logaritmica(señal_audio)



