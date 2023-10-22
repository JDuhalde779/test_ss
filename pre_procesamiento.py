import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
import sounddevice as sd
import time

t=10
# Ver la lista de dispositivos de audio
#devices = sd.query_devices()#
# print(devices) #
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
    sine_sweep = generar_sine_sweep_y_inversa(duracion, frec_comienzo, frec_final, fs=44100)
    
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

def plot_dominio_temporal(señal, fs= 44100):
    """
    Muestra el dominio temporal de la señal.

    Parametros
    ----------
    señal : NumPy array
        Señal a mostrar en el dominio temporal.
    fs : int
        Frecuencia de muestreo en Hz de la señal.

    Returns
    -------
    None
    """
    # Calcula los valores de tiempo
    tiempo = np.arange(len(señal)) / fs

    # Crea una nueva figura y plotea la señal
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, señal)
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

def generar_sine_sweep_y_inversa(duracion, frec_comienzo, freq_final, fs=44100):
    """
    Genera un sine sweep with logaritmico y su correspondiente filtro inverso.

    Parametros:
    duracion (float): Duracion del Sine Sweep en segundos.
    frec_comienzo(float):frequencia de comienzo en Hz.
    freq_final (float): frecuencia final en Hz.
    fs (int):frecuencia de sampleo en Hz. Predeterminado en 44100 Hz.

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


    t_swipe_arange = np.arange(0, duracion*fs)/fs  # Arreglo de muestreos
    R = np.log(freq_final/frec_comienzo)  # Ratio del Sweep
    K = duracion*2*np.pi*frec_comienzo/R
    L = duracion/R
    w = (K/L)*np.exp(t_swipe_arange/L)
    m = frec_comienzo/w

    # Calcula el filtro inverso  k(t)
    x_t = sine_sweep
    k_t = m * x_t[::-1]  #  Inversion temporal de x(t)

    # Normaliza el Filtro Inverso 
    k_t /= np.max(np.abs(k_t))
    
    # Calcula el eje de tiempo  en segundos
    tiempo_segundos = np.linspace(0, duracion, len(sine_sweep), endpoint=False)    

    # Plotea el Sine Sweep generado
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(tiempo_segundos, sine_sweep)
    plt.title('Sine Sweep logaritmico')
    plt.xlabel('Tiempo (seconds)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    

test_ruido_rosa(t)