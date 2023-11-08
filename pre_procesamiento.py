import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
import sounddevice as sd
import time
import scipy.io.wavfile as wav

t=10
# Ver la lista de dispositivos de audio
#devices = sd.query_devices()
#print(devices) 
def test_ruido_rosa(t,ncols=16,fs=44100):
    duracion = 5  # Duración del sine sweep en segundos
    frec_comienzo = 20  # Frecuencia inicial en Hz
    frec_final = 20000  # Frecuencia final en Hz
    fs = 44100
    

    # Genera el ruido rosa y lo almacena en 'audio'
    audio = ruidoRosa_voss(t, ncols=16, fs=44100)
    plt.plot(audio)
    plt.show()
    dominio = plot_dominio_temporal(audio, 44100)
    

    # Genera el Sine Sweep
    sine_sweep = generar_sine_sweep_y_inversa(duracion, frec_comienzo, frec_final, fs=44100, periodo=1.0)
    
    # Guarda el Sine Sweep como archivo de audio .wav
    sf.write('sine_sweep_log.wav', sine_sweep, fs)

    señal = 'sine_sweep_log.wav'
    
    grabar_señal(señal, 1, 2, 5)


   
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



# Parámetros del sine sweep logarítmico
duracion = 5  # Duración en segundos
frec_comienzo = 20  # Frecuencia inicial en Hz
freq_final = 20000  # Frecuencia final en Hz
fs = 44100  # Frecuencia de muestreo en Hz

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


