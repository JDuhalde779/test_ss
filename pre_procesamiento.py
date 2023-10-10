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
    duration = 5  # Duración del sine sweep en segundos
    start_freq = 20  # Frecuencia inicial en Hz
    end_freq = 20000  # Frecuencia final en Hz
    fs = 44100
    

    # Genera el ruido rosa y lo almacena en 'audio'
    audio = ruidoRosa_voss(t, ncols=16, fs=44100)
    plt.plot(audio)
    plt.show()
    dominio = plot_temporal_domain(audio, 44100)
    plot_frequency_response(audio, fs=44100)

    # Genera el Sine Sweep
    sine_sweep = generate_log_sine_sweep_and_inverse(duration, start_freq, end_freq, fs=44100)
    
    # Guarda el Sine Sweep como archivo de audio .wav
    sf.write('sine_sweep_log.wav', sine_sweep, fs)

    signal = 'sine_sweep_log.wav'
    # Registra el tiempo de inicio
    inicio_grabacion = time.time()

    # Llamar a la función para grabar y reproducir audio
    record_signal(signal, 25, 25, 5)

    # Registra el tiempo de finalización
    fin_reproduccion = time.time()

    # Calcula la latencia en segundos
    latencia_segundos = fin_reproduccion - inicio_grabacion

    # Convierte la latencia a milisegundos
    latencia_milisegundos = latencia_segundos * 1000

    print(f'Latencia de la función grabar_reproducir_audio: {round(latencia_milisegundos, 2)} ms')

   
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

def plot_temporal_domain(signal, fs= 44100):
    """
    Muestra el dominio temporal de la señal.

    Parametros
    ----------
    signal : NumPy array
        Señal a mostrar en el dominio temporal.
    fs : int
        Frecuencia de muestreo en Hz de la señal.

    Returns
    -------
    None
    """
    # Calculate the time values
    time = np.arange(len(signal)) / fs

    # Create a new figure and plot the signal
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.title('Dominio Temporal de la Señal')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.show()


def plot_frequency_response(signal, fs=44100):
    """
    Plots the frequency response of a signal in decibels (dB).

    Parameters:
    signal (numpy array): The input signal.
    fs (int): The sampling frequency of the signal.

    Returns:
    None
    """
    f, Pxx = welch(signal, fs=44100, nperseg=44100)
    plt.figure(figsize=(10, 4))
    plt.semilogx(f, 10 * np.log10(Pxx))
    plt.title('Frequency Response of the Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.xlim([20, 20000])
    plt.show()



# Parámetros del sine sweep logarítmico
duration = 5  # Duración en segundos
start_freq = 20  # Frecuencia inicial en Hz
end_freq = 20000  # Frecuencia final en Hz
fs = 44100  # Frecuencia de muestreo en Hz

def generate_log_sine_sweep_and_inverse(duration, start_freq, end_freq, fs=44100):
    """
    Generate a logarithmic sine sweep with a 3 dB/octave increase in amplitude
    and its corresponding inverse filter.

    Parameters:
    duration (float): Duration of the sweep in seconds.
    start_freq (float): Starting frequency in Hz.
    end_freq (float): Ending frequency in Hz.
    fs (int): Sampling frequency in Hz. Default is 44100 Hz.

    Returns:
    tuple: A tuple containing the generated logarithmic sine sweep and its inverse filter.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    omega_start = 2 * np.pi * start_freq
    omega_end = 2 * np.pi * end_freq

    # Calculate the desired amplitude change per octave (3 dB/octave)
    amplitude_change_per_octave = 3.0  # 3 dB/octave

    # Calculate the amplitude scaling factor for each frequency
    freqs = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), len(t)))
    amplitude_scale = 10 ** (amplitude_change_per_octave * np.log2(freqs / start_freq) / 20)

    # Generate the sine sweep with scaling
    sine_sweep = np.sin(np.cumsum(2 * np.pi * freqs / fs))
    sine_sweep *= amplitude_scale

    # Normalize the sine sweep
    sine_sweep /= np.max(np.abs(sine_sweep))

    # Modulate w(t)
    K = end_freq - start_freq
    L = duration
    w_t = (K / L) * np.exp(t / L)

    # Calculate the instantaneous frequency m(t)
    m_t = (start_freq / (2 * np.pi)) / w_t

    # Calculate the inverse filter k(t)
    x_t = sine_sweep
    k_t = m_t * x_t[::-1]  # Temporal inversion of x(t)

    # Normalize the inverse filter
    k_t /= np.max(np.abs(k_t))
    
    # Calculate the time axis in seconds
    time_seconds = np.linspace(0, duration, len(sine_sweep), endpoint=False)    

    # Plot the generated sine sweep
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(time_seconds, sine_sweep)
    plt.title('Logarithmic Sine Sweep with 3 dB/octave Increase')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot the inverse filter
    plt.figure(figsize=(10, 4))
    plt.plot(t, k_t)
    plt.title('Inverse Filter k(t)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
    # Guarda el filtro inverso k(t) como archivo de audio .wav
    sf.write('filtro_inverso.wav', k_t, fs)
   
    return sine_sweep

def record_signal(signal, input_device, output_device, duration):
    """
    Reproducción y grabación de una señal en formato ".wav" en simultáneo.

    Parámetros
    ----------
    signal: Archivo ".wav"

    input_device: int
        Dispositivo de grabación de audio.
    
    output_device: int
        Dispositivo de reproducción de audio.

    duration: 
        Tiempo de grabación de la señal.

    Para ver el listado de dispositivos de audio: 
    
    import sounddevice as sd
    sd.query_devices()
        
    Ejemplo
    -------
    import numpy as np
    import soundfile as sf
    import sounddevice as sd
    
    signal = 'SineSweepLog.wav'
    input_device = 1
    output_device = 9
    record_signal(signal, input_device, output_device)
    
    """
    
    # Selección de dispositivos de audio
    sd.default.device = input_device, output_device
    # Reproducción de la señal y grabación en simultáneo   
    data, fs = sf.read(signal, dtype='float32')
    samples_rec = duration*fs
    val = data[0:samples_rec]
    signal_recording = sd.playrec(val, fs, channels=1)
    sd.wait()
    sf.write('signal_recording.wav', signal_recording,fs )  # Guardo el archivo .wav
    

test_ruido_rosa(t)