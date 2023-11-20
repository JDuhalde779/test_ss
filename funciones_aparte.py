import os
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from pydub.playback import play
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import welch
import sounddevice as sd
import time
import scipy.io.wavfile as wav
import scipy.signal as signal

def grabar_señal(señal, disp_entrada, disp_salida, duracion):
    """
    Reproducción y grabación de una señal en formato ".wav" en simultáneo.

    Parámetros
    ----------
    signal: Archivo ".wav"

    disp_entrada: int
        Dispositivo de grabación de audio.
    
    disp_salida: int
        Dispositivo de reproducción de audio.

    duracion: 
        Tiempo de grabación de la señal.

    Para ver el listado de dispositivos de audio: 
    
    import sounddevice as sd
    sd.query_devices()
        
    Ejemplo
    -------
    import numpy as np
    import soundfile as sf
    import sounddevice as sd
    
    señal = 'SineSweepLog.wav'
    disp_entrada = 1
    disp_salida = 9
    grabar_señal(señal, disp_entrada, disp_salida)
    
    """
    
    # Selección de dispositivos de audio
    sd.default.device = disp_entrada, disp_salida
    # Reproducción de la señal y grabación en simultáneo   
    data, fs = sf.read(señal, dtype='float32')
    samples_rec = duracion*fs
    val = data[0:samples_rec]
    inicio = time.time()
    grabacion_señal = sd.playrec(val, fs, channels=1)
    sd.wait()
    final = time.time() 
    latencia = final - inicio
    latencia_real = latencia - duracion
    print("Latencia: ", latencia_real)
    sf.write('signal_recording.wav', grabacion_señal,fs )  # Guardo el archivo .wav

    return grabacion_señal


señal = 'sine_sweep_log.wav'

grabar_señal(señal,8,8,5)




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

señal = sf.read("1st_baptist_nashville_balcony_mono_copy.wav")
#plot = plot_dominio_temporal(señal)

def acortar_wav(input_path, output_path, duracion_deseada):
    # Cargar el archivo WAV
    audio = AudioSegment.from_wav(input_path)

    # Acortar la duración según lo deseado
    audio_recortado = audio[:duracion_deseada * 1000]  # Duración en milisegundos

    # Guardar el nuevo archivo WAV
    audio_recortado.export(output_path, format="wav")

# Especifica la ruta del archivo de entrada y salida, y la nueva duración deseada en segundos
archivo_entrada = "salida_filtrada_1000_fpm.wav"
archivo_salida = "impulso_recortado.wav"
duracion_deseada = 2  # Por ejemplo, 10 segundos    
acortar_wav(archivo_entrada,archivo_salida,duracion_deseada)


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







