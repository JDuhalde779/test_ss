import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
t=10
def test_ruido_rosa(t,ncols=16,fs=44100):
    audio = ruidoRosa_voss(t,ncols=16,fs=44100)
    plt.plot(audio)
    plt.show()
    dominio = plot_temporal_domain(audio,44100)

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
    Generar un `.wav` desde un numpy array de 10 segundos con ruido rosa a una 
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
    
if __name__ == '__main__':
    print(test_ruido_rosa(t))