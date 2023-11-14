import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import wave
import scipy.signal as signal
from pydub import AudioSegment
from scipy.signal import welch
from scipy.signal import find_peaks
from pydub.playback import play
import sounddevice as sd
import time
import scipy.io.wavfile as wav
from scipy.fft import fft, ifft
from funciones_aparte import plot_dominio_temporal
from funciones_aparte import stereo_a_mono_wav
from funciones_aparte import iec61260_filtros
from funciones_aparte import acortar_wav

t=10
duracion = 5  # Duración del sine sweep en segundos
frec_comienzo = 20  # Frecuencia inicial en Hz
frec_final = 20000  # Frecuencia final en Hz
fs = 44100
    
def ruidoRosa_voss(t,ncols=16,fs=44100):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney(https://www.dsprelated.com/showabstract/3933.php).
    
    .. Nota:: si 'ruidoRosa.wav' existe, este será sobreescrito
    
    Parametros
    ----------
    t : float
        Valor temporal en segundos, este determina la duración del ruido generado.
    rcols: int
        Determina el número de fuentes a aleatorias a agregar.
    fs: int
        Frecuencia de muestreo en Hz de la señal. Por defecto el valor es 44100 Hz.
    
    returns: NumPy array
        Datos de la señal generada.
    
    Ejemplo
    -------
    Generar un .wav desde un numpy array de 10 segundos con ruido rosa a una 
    frecuencia de muestreo de 44100 Hz.
    
        import numpy as np
        import soundfile as sf
        from scipy import signal
        
        ruidoRosa_voss(10)
    """
    nrows=int(t*fs)
    array = np.full((nrows, ncols), np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # el numero total de cambios es nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)
    
    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)
    
    ## Centrado de el array en 0
    total = total - total.mean()
    
    ## Normalizado
    valor_max = max(abs(max(total)),abs(min(total)))
    total = total / valor_max
    
    # Agregar generación de archivo de audio .wav
    sf.write('ruidoRosa.wav', total, fs)
    
    ## Plot de total en cantidad de muestras (t*fs)
    plt.plot(total)

    
    return total

# Genera el ruido rosa y lo almacena en 'audio'
audio = ruidoRosa_voss(t, ncols=16, fs=44100)
plt.plot(audio)
plt.show()
dominio = plot_dominio_temporal(audio, 44100)

def generar_sine_sweep_y_inversa(duracion, frec_comienzo, freq_final, fs=44100, periodo=1.0):
    """
    Genera un sine sweep logaritmico y su correspondiente filtro inverso.

    Parametros:
    duracion (float): Duracion del Sine Sweep en segundos.
    frec_comienzo(float):frequencia de comienzo en Hz.
    freq_final (float): frecuencia final en Hz.
    fs (int):frecuencia de sampleo en Hz. Predeterminado en 44100 Hz.
    periodo (float): Duración del período a graficar en segundos. Predeterminado en 1.0 segundo.

    Returns:
    tuple: Una tupla que contiene el sine sweep logaritmico generado
    """
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    
    # Calcula el escalamiento de amplitud para cada fercuencia
    freqs = np.exp(np.linspace(np.log(frec_comienzo), np.log(freq_final), len(t)))

    # Genera el Sine Sweep 
    sine_sweep = np.sin(np.cumsum(2 * np.pi * freqs / fs))

    # Normaliza el Sine Sweep
    sine_sweep /= np.max(np.abs(sine_sweep))

    # Encuentra el índice que corresponde al final del período a graficar
    periodo_samples = int(periodo * fs)
    
    # Gráfica solo un período del Sine Sweep
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(t[:periodo_samples], sine_sweep[:periodo_samples])
    plt.title('Sine Sweep logaritmico (1 periodo)')
    plt.xlabel('Tiempo (seconds)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calcula el filtro inverso
    t_swipe_arange = np.arange(0, duracion*fs)/fs
    R = np.log(freq_final/frec_comienzo)
    K = duracion*2*np.pi*frec_comienzo/R
    L = duracion/R
    w = (K/L)*np.exp(t_swipe_arange/L)
    m = frec_comienzo/w
    x_t = sine_sweep
    k_t = m * x_t[::-1]
    k_t /= np.max(np.abs(k_t))
    
     # Plotea el Filtro Inverso 
    plt.figure(figsize=(10, 4))
    plt.plot(t, k_t)
    plt.title('Filtro Inverso k(t)')
    plt.xlabel('Tiempo (seconds)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()
    

    # Guarda el filtro inverso k(t) como archivo de audio .wav
    sf.write('filtro_inverso.wav', k_t, fs)
   
    return sine_sweep
    

# Genera el Sine Sweep
sine_sweep = generar_sine_sweep_y_inversa(duracion, frec_comienzo, frec_final, fs=44100, periodo=1.0)   
# Guarda el Sine Sweep como archivo de audio .wav
sf.write('sine_sweep_log.wav', sine_sweep, fs)

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
input_file = 'sine_sweep_log.wav'  # Nombre del archivo .wav del sine sweep
output_file = 'filtro_inversoGENERADO.wav'  # Nombre del archivo de salida del filtro inverso

generar_filtro_inverso(input_file, output_file)

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
    grabacion_señal = sd.playrec(val, fs, channels=1)
    sd.wait()
    sf.write('signal_recording.wav', grabacion_señal,fs )  # Guardo el archivo .wav
    return grabar_señal


señal = 'sine_sweep_log.wav'   
grabar_señal(señal, 1, 2, 5)


   
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


stereo_a_mono_wav(archivo_seleccionado)


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


#Reproducir la respuesta al impulso generada
#play(AudioSegment.from_wav(nombre_archivo))

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
sine_sweep_wav = 'sine_sweep_log.wav'  # Archivo .wav del sine sweep logarítmico
filtro_inverso_wav = 'filtro_inversoGENERADO.wav'  # Archivo .wav del filtro inverso
salida_wav = 'respuesta_al_impulsoGENERADO.wav'  # Nombre del archivo de salida de la respuesta al impulso

respuesta_al_impulso(sine_sweep_wav, filtro_inverso_wav, salida_wav)

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

    # Aplicar la conversión logarítmica
    audio_log = 20 * np.log10(np.abs(audio_data + epsilon))

    return audio_log


señal_audio = "1st_baptist_nashville_balcony_mono_copy.wav"
audio_log = convertir_audio_a_escala_logaritmica(señal_audio)


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
audio_signal, sample_rate = sf.read("1st_baptist_nashville_balcony_mono_copy.wav")
frecuencias_centrales = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
for i in frecuencias_centrales:
    iec61260_filtros(audio_signal,i, sample_rate=44100)



# Especifica la ruta del archivo de entrada y salida, y la nueva duración deseada en segundos
archivo_entrada = "respuesta_al_impulsoGENERADO.wav"
archivo_salida = "impulso_recortado.wav"
duracion_deseada = 5  # Por ejemplo, 10 segundos
acortar_wav(archivo_entrada, archivo_salida, duracion_deseada)


def filtro_promedio_movil(input_file, output_file, L):
    # Leer el archivo WAV de entrada
    sample_rate, audio_data = wav.read(input_file)
   
    # Aplicar el filtro de promedio móvil
    filtered_signal = np.zeros_like(audio_data, dtype=np.float64)

    for i in range(L, len(audio_data)):
        filtered_signal[i] = (1/L) * np.sum(audio_data[i-L+1:i+1])

    # Guardar la señal filtrada en un archivo WAV de salida
    wav.write(output_file, sample_rate, filtered_signal.astype(np.int16))


input_file = "impulso_recortado.wav"
output_file = "salida_filtrada.wav"
L = 100 # Número de muestras para el promedio móvil
filtro_promedio_movil(input_file, output_file, L)


def calculate_schroeder_integral(hA):
    """
    Calcula la integral de Schroeder para una respuesta al impulso dada.
    Parameters:
    - hA: np.array, respuesta al impulso suavizada.
    Returns:
    - E: np.array, valores de la integral de Schroeder.
    """
    # Verificar que no haya valores nan o inf en la respuesta al impulso
    if np.isnan(hA).any() or np.isinf(hA).any():
        raise ValueError("La respuesta al impulso contiene valores no numéricos.")
    # Crear el arreglo de tiempo
    tau = np.arange(0, len(hA))
    # Calcular la integral de Schroeder
    E = np.sum(hA[:]**2) - np.cumsum(hA[:]**2)
    E = 10 * np.log10(E / np.sum(hA**2))
    return E

# Calcular la integral de Schroeder
hA, tasa_muestreo = sf.read("salida_filtrada.wav")
integral_schroeder = calculate_schroeder_integral(hA)
sf.write("Audio_Schroeder.wav", integral_schroeder, 44100)
# Imprimir el resultado
print(integral_schroeder)


