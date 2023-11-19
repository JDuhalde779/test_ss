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
   

    
    return total


# Genera el ruido rosa y lo almacena en 'audio'
ruido_rosa = ruidoRosa_voss(t, ncols=16, fs=44100)
def normalizado_audio(ruido_rosa):
     
    ## Normalizado
    valor_max = max(abs(max(ruido_rosa)),abs(min(ruido_rosa)))
    ruido_rosa = ruido_rosa / valor_max
    
    # Agregar generación de archivo de audio .wav
    sf.write('ruidoRosa.wav', ruido_rosa, fs)
    
    ## Plot de total en cantidad de muestras (t*fs)
    plt.plot(ruido_rosa)

    return ruido_rosa

audio = normalizado_audio(ruido_rosa)
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

#VA ACA

   
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


señal_audio = "concert_hall_york_university\\rir_jack_lyons_lp1_96k_mono.wav"
audio_log = convertir_audio_a_escala_logaritmica(señal_audio)


def iec61260_filtros(audio_signal, center_frequency, tipo_de_filtro, sample_rate=44100):
    # Lista de frecuencias centrales según la norma IEC61260 para octavas y tercios de octava
    center_frequencies = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    frecuencias_centrales_tercio = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]

    if tipo_de_filtro == "octava":
        if center_frequency in center_frequencies:
            G = 1.0 / 2.0
            factor = np.power(2, G)
            center_frequency_hz = center_frequency

            lower_cutoff_frequency_hz = center_frequency_hz / factor
            upper_cutoff_frequency_hz = center_frequency_hz * factor

            sos = signal.iirfilter(4, [lower_cutoff_frequency_hz, upper_cutoff_frequency_hz],
                                   rs=60, btype='band', analog=False,
                                   ftype='butter', fs=sample_rate, output='sos')
            filtered_signal = signal.sosfilt(sos, audio_signal)

            # Devolver la señal filtrada
            return filtered_signal

        else:
            print("Se ha ingresado un valor de frecuencia inválido")
            return None

    if tipo_de_filtro == "tercio":
        if center_frequency in frecuencias_centrales_tercio:
            G = 1.0 / 6.0
            factor = np.power(2, G)
            center_frequency_hz = center_frequency

            lower_cutoff_frequency_hz = center_frequency_hz / factor
            upper_cutoff_frequency_hz = center_frequency_hz * factor

            sos = signal.iirfilter(4, [lower_cutoff_frequency_hz, upper_cutoff_frequency_hz],
                                   rs=60, btype='band', analog=False,
                                   ftype='butter', fs=sample_rate, output='sos')
            filtered_signal = signal.sosfilt(sos, audio_signal)

            # Devolver la señal filtrada
            return filtered_signal

        else:
            print("Se ha ingresado un valor de frecuencia inválido")
            return None



#for i in frecuencias_centrales: #Se puede modificar en caso que se desee el filtro en tercio de octavas
#    iec61260_filtros(audio_signal,i, sample_rate)

def filtro_promedio_movil(input_signal, output_file, L, sample_rate):
    # Aplicar el filtro de promedio móvil
    filtered_signal = np.zeros_like(input_signal, dtype=np.float64)

    for i in range(L, len(input_signal)):
        filtered_signal[i] = (1/L) * np.sum(input_signal[i-L+1:i+1])

    # Guardar la señal filtrada en un archivo WAV de salida
    sf.write(output_file, filtered_signal, sample_rate)


# Llamar a la función con alguna RI generada anteriormente.
#audio_signal, sample_rate = sf.read('\Users\Educacion\Desktop\Test funcion filtrosruidoRosa.wav')
audio_signal, sample_rate = sf.read("concert_hall_york_university\\rir_jack_lyons_lp1_96k_mono.wav")
frecuencias_centrales = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
frecuencias_centrales_tercio = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
tipo_de_filtro = "octava"
# Crear un diccionario para almacenar las señales filtradas
signals = {}
print (signals)

# Iterar sobre las frecuencias centrales y aplicar los filtros
for center_frequency in frecuencias_centrales:
    filtered_signal = iec61260_filtros(audio_signal, center_frequency, tipo_de_filtro, sample_rate)
    signals[center_frequency] = filtered_signal

# Aplicar el filtro de promedio móvil y guardar las señales filtradas
L = 100  # Número de muestras para el promedio móvil

for center_frequency, filtered_signal in signals.items():
    output_file = f"salida_filtrada_{center_frequency}_fpm.wav"
    filtro_promedio_movil(filtered_signal, output_file, L, sample_rate)



# Especifica la ruta del archivo de entrada y salida, y la nueva duración deseada en segundos
#archivo_entrada = "1st_baptist_nashville_balcony_mono copy.wav"
#archivo_salida = "impulso_recortado.wav"
#duracion_deseada = 5  # Por ejemplo, 10 segundos
# acortar_wav(archivo_entrada, archivo_salida, duracion_deseada)

""""
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
#filtro_promedio_movil(input_file, output_file, L)
"""

epsilon=1e-10
def calcular_schroeder_integral(p_t, lim=4):
    """
    Calcula la integral de Schroeder.
    Parameters:
    - p_t: np.array, respuesta al impulso suavizada.
    Returns:
    numpy.ndarray: El array de la integral de Schroeder.
    """
    # Verificar que no haya valores nan o inf en la respuesta al impulso
    if np.isnan(p_t).any() or np.isinf(p_t).any():
        raise ValueError("La respuesta al impulso contiene valores no numéricos.")
    cut_lim = int(lim*fs)
    print("cut_lim=",cut_lim)
    # Calcular la integral de Schroeder
    E = np.sum(p_t[:]**2) - np.cumsum(p_t[:cut_lim]**2)
    E = 10 * np.log10(E / np.sum(p_t**2))
    #E = 10* np.log10(np.sum(p_t[:]**2) - np.cumsum(p_t[:cut_lim]**2))
    #E = 10 * np.log10(np.cumsum(p_t[cut_lim::-1]**2)[::-1] / np.sum(p_t**2))
    E = E[:len(p_t)]

    return E

# Calcular la integral de Schroeder
p_t, fs = sf.read("señal_filtrada_1000.wav")
print ("longitud de p_t",len(p_t))
lim = int((len(p_t))/fs)
integral_schroeder = calcular_schroeder_integral(p_t)
print("longitud schroeder", len(integral_schroeder))
sf.write("Audio_Schroeder.wav", integral_schroeder, 44100)
# Imprimir el resultado
print("longitud de la señal:", len(p_t))
print("longitud de schroeder", len(integral_schroeder))
print(integral_schroeder)


"""
# Crear el arreglo de tiempo
tau = np.arange(0, len(signal))
# Graficar la señal original y la integral de Schroeder
plt.subplot(2, 1, 1)
plt.plot(tau, signal, label='Respuesta al Impulso Suavizada')
plt.title('Señal Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(tau, integral_schroeder, label='Integral de Schroeder')
plt.title('Integral de Schroeder')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud (dB)')
plt.legend()
plt.tight_layout()
plt.show()

"""



def plot_dominio_temporal2(señal1, señal2, fs=44100, inicio=None, duracion=None, umbral_amplitud=None):
    """
    Muestra el dominio temporal de la señal con un umbral de amplitud.

    Parámetros
    ----------
    señal : np.array
        Array que representa la señal de audio.
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
    # Calcula los valores de tiempo
    tiempo = np.arange(len(señal1)) / fs
    tiempo2 = np.arange(len(señal2)) / fs

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
    fin_idx = min(len(señal1), fin_idx)
    fin_idx2 = min(len(señal2), fin_idx)

    # Aplicar umbral de amplitud si se proporciona
    if umbral_amplitud is not None:
        señal1[señal1 < umbral_amplitud] = umbral_amplitud

    # Crea una nueva figura y plotea la señal en la ventana especificada
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo[inicio_idx:fin_idx], señal1[inicio_idx:fin_idx])
    plt.plot(tiempo[inicio_idx:fin_idx2], señal2[inicio_idx:fin_idx2])
    plt.title('Dominio Temporal de la Señal')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()




def convertir_audio_a_escala_logaritmica2(señal_audio):
    """
    Convierte un archivo de audio en escala logarítmica y devuelve el resultado como un array.

    Parámetros:
    señal_audio (np.array): Array de la señal de audio.

    Retorna:
    numpy.ndarray: El array de la señal en escala logarítmica.
    """
    # Normalizar los valores de audio entre -1 y 1
    audio_data = señal_audio.astype(np.float32) / np.max(np.abs(señal_audio))

    # Aplicar la conversión logarítmica
    audio_log = 20 * np.log10(np.abs(audio_data))

    return audio_log
# Normaliza la señal
normalized_audio = convertir_audio_a_escala_logaritmica2(p_t)
plot_dominio_temporal2(normalized_audio, integral_schroeder, fs=44100)

def calcular_edt(schroeder, fs):
    """
    Calcula el tiempo de reverberación EDT desde una función de Schroeder suavizada.

    Parámetros:
    - schroeder: np.array, función de Schroeder suavizada.
    - fs: int, frecuencia de muestreo en Hz.

    Retorna:
    - edt: float, tiempo de reverberación EDT en segundos.
    """
    

    # Encuentra el tiempo en el que la integral cruza -10 dB
    umbral = -10  # Puedes ajustar este valor según las especificaciones de la ISO 3382
    indice_cruce = np.where(schroeder <= umbral)[0]
    print("Indice de cruce =", indice_cruce)
    if indice_cruce.size > 0:
        # Toma el primer índice de cruce
        indice_cruce = indice_cruce[0]

        # Convierte el índice a tiempo en segundos
        tiempo_cruce = indice_cruce / fs

        print("Tiempo de cruce =", tiempo_cruce)

        # El EDT es seis veces el tiempo en que la integral cruza -10 dB
        edt = 6 * tiempo_cruce
    
    else:
        # No se encontró ningún índice que cumpla con el umbral
        edt = 0  # O cualquier otro valor que desees asignar

    return edt
# Ejemplo de uso:
# Supongamos que tienes la función de Schroeder suavizada y la frecuencia de muestreo
schroeder = integral_schroeder  # Reemplaza con tu función de Schroeder
fs = 44100  # Reemplaza con tu frecuencia de muestreo
# Calculate the EDT
edt = calcular_edt(integral_schroeder, fs)
# Imprime el resultado
print("EDT:", edt, "segundos")

def calcular_t10(schroeder, fs):
    """
    Calcula el tiempo de reverberación T10 desde una función de Schroeder suavizada.

    Parámetros:
    - schroeder: np.array, función de Schroeder suavizada.
    - fs: int, frecuencia de muestreo en Hz.

    Retorna:
    - t10: float, tiempo de reverberación T10 en segundos.
    """
    # Encuentra el tiempo en el que la integral cruza -5 dB
    umbral_A= -5  # Puedes ajustar este valor según las especificaciones de la ISO 3382
    umbral_B = -15
    indice_cruce_5 = np.where(schroeder <= umbral_A)[0]
    indice_cruce_15 = np.where(schroeder <= umbral_B)[0]

    # Toma el primer índice de cruce
    indice_cruce_5 = indice_cruce_5[0]
    indice_cruce_15 = indice_cruce_15[0]
    
    # Convierte el índice a tiempo en segundos
    tiempo_cruce_5 = indice_cruce_5 / fs
    tiempo_cruce_15 = indice_cruce_15 / fs
    # La diferencia entre los dos cruces determina el tiempo de reverberación T10
    t10 = tiempo_cruce_15 - tiempo_cruce_5
    return t10

# Supongamos que tienes la función de Schroeder suavizada y la frecuencia de muestreo
schroeder = integral_schroeder  # Reemplaza con tu función de Schroeder
fs = 44100  # Reemplaza con tu frecuencia de muestreo
# Calculate the EDT
t10 = calcular_t10(integral_schroeder, fs)
# Imprime el resultado
print("T10:", t10, "segundos")

def calcular_t20(schroeder, fs):
    """
    Calcula el tiempo de reverberación T20 desde una función de Schroeder suavizada.

    Parámetros:
    - schroeder: np.array, función de Schroeder suavizada.
    - fs: int, frecuencia de muestreo en Hz.

    Retorna:
    - t10: float, tiempo de reverberación T10 en segundos.
    """
    # Encuentra el tiempo en el que la integral cruza -5 dB
    umbral_A= -5  # Puedes ajustar este valor según las especificaciones de la ISO 3382
    umbral_B = -25
    indice_cruce_5 = np.where(schroeder <= umbral_A)[0]
    indice_cruce_25 = np.where(schroeder <= umbral_B)[0]

    # Toma el primer índice de cruce
    indice_cruce_5 = indice_cruce_5[0]
    indice_cruce_25 = indice_cruce_25[0]
    
    # Convierte el índice a tiempo en segundos
    tiempo_cruce_5 = indice_cruce_5 / fs
    tiempo_cruce_25 = indice_cruce_25 / fs
    # La diferencia entre los dos cruces determina el tiempo de reverberación T10
    t20 = tiempo_cruce_25 - tiempo_cruce_5
    return t20

# Supongamos que tienes la función de Schroeder suavizada y la frecuencia de muestreo
schroeder = integral_schroeder  # Reemplaza con tu función de Schroeder
fs = 44100  # Reemplaza con tu frecuencia de muestreo
# Calculate the EDT
t20 = calcular_t20(integral_schroeder, fs)
# Imprime el resultado
print("T20:", t20, "segundos")

def calcular_t30(schroeder, fs):
    """
    Calcula el tiempo de reverberación T30 desde una función de Schroeder suavizada.

    Parámetros:
    - schroeder: np.array, función de Schroeder suavizada.
    - fs: int, frecuencia de muestreo en Hz.

    Retorna:
    - t30: float, tiempo de reverberación T30 en segundos.
    """
    # Encuentra el tiempo en el que la integral cruza -5 dB
    umbral_A= -5  # Puedes ajustar este valor según las especificaciones de la ISO 3382
    umbral_B = -35
    indice_cruce_5 = np.where(schroeder <= umbral_A)[0]
    indice_cruce_35 = np.where(schroeder <= umbral_B)[0]

    # Toma el primer índice de cruce
    indice_cruce_5 = indice_cruce_5[0]
    indice_cruce_35 = indice_cruce_35[0]
    
    # Convierte el índice a tiempo en segundos
    tiempo_cruce_5 = indice_cruce_5 / fs
    tiempo_cruce_35 = indice_cruce_35 / fs
    # La diferencia entre los dos cruces determina el tiempo de reverberación T10
    t30 = tiempo_cruce_35 - tiempo_cruce_5
    return t30

# Supongamos que tienes la función de Schroeder suavizada y la frecuencia de muestreo
schroeder = integral_schroeder  # Reemplaza con tu función de Schroeder
fs = 44100  # Reemplaza con tu frecuencia de muestreo
# Calcular T30
t30 = calcular_t30(integral_schroeder, fs)
# Imprime el resultado
print("T30:", t30, "segundos")



def calculate_d50(schroeder, fs):
    """
    Calcula el parámetro D50 desde una función de Schroeder suavizada.

    Parámetros:
    - schroeder: np.array, función de Schroeder suavizada.
    - fs: int, frecuencia de muestreo en Hz.

    Retorna:
    - d50: float, parámetro D50.
    """
    # Definir los límites de las integrales
    t_0_05 = 0.05  # límite superior para la primera integral
    t_inf = len(schroeder) / fs  # límite superior para la segunda integral

    # Convertir los tiempos a índices
    idx_0_05 = int(t_0_05 * fs)
    idx_inf = len(schroeder)

    print("Indices:", idx_0_05, idx_inf)

    # Calcular las dos integrales utilizando np.sum y manejar NaN
    integral_0_05 = np.sum(schroeder[:idx_0_05]**2) 
    integral_inf = np.sum(np.nan_to_num(schroeder[idx_0_05:idx_inf])**2) 

    print("Integrales:", integral_0_05,";" ,integral_inf)

    # Calcular el parámetro D50
    d50 = integral_0_05 / integral_inf

    return d50

# Ejemplo de uso:
# Supongamos que tienes la función de Schroeder suavizada y la frecuencia de muestreo
schroeder =  integral_schroeder 
fs = 44100  # Reemplaza con tu frecuencia de muestreo
# Calcular D50
d50 = calculate_d50(integral_schroeder, fs)
# Imprimir el resultado
print("D50:", d50, "%")



def calculate_c80(integral_schroeder, fs, t=0.08):
    """
    Calcula el C80 a partir de la integral de Schroeder.
    Parameters:
    integral_schroeder (np.array): Array de la integral de Schroeder.
    fs (int): Frecuencia de muestreo de la señal de audio.
    t (float): Tiempo en segundos para el cálculo del C80 (por defecto, 80 ms).
    Returns:
    float: Valor de C80 en decibelios.
    """
    # Calcula el índice correspondiente al tiempo t
    t_index = int(t * fs)

    # Calcula la energía sonora temprana y tardía
    early_energy = np.sum(integral_schroeder[:t_index])
    late_energy = np.sum(integral_schroeder[t_index:])

    # Calcula el C80 en decibelios
    c80 = 10 * np.log10(early_energy / late_energy +  epsilon)

    return c80

# Llamada a la función para calcular el C80
c80_value = calculate_c80(integral_schroeder, fs)

# Imprimir el resultado
print("C80:", c80_value)

