import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import soundfile as sf
def test_ruido_rosa(nrows,ncols):
    audio = ruidoRosa_voss(nrows,ncols)
    plt.plot(audio)
    plt.show()
    dominio = plot_temporal_domain(audio,100)
    
tiempo = 10  

def ruidoRosa_voss(duration, fs=44100):
    """
    Genera ruido rosa utilizando el algoritmo de Voss-McCartney.

    Parametros
    ----------
    duration : float
        Duración en segundos del ruido rosa a generar.
    fs : int, optional
        Frecuencia de muestreo en Hz de la señal. Por defecto, el valor es 44100 Hz.

    Returns
    -------
    NumPy array
        Datos de la señal generada.

    Ejemplo
    -------
    Generar un archivo ".wav" de 10 segundos con ruido rosa a una frecuencia de muestreo de 44100 Hz.

        import numpy as np
        import soundfile as sf
        
        ruidoRosa_voss(10)
    """
    
    # Calculate the number of samples based on duration and sample rate
    num_samples = int(duration * fs)

    array = np.full((num_samples, 1), np.nan)
    array[0, 0] = np.random.random()
    
    # The number of changes is equal to the number of samples
    n = num_samples
    cols = np.random.geometric(0.5, n)
    cols[cols >= 1] = 0
    rows = np.random.randint(num_samples, size=n)
    array[rows, cols] = np.random.random(n)
    
    df = pd.DataFrame(array)
    filled = df.fillna(method='ffill', axis=0)
    total = filled.sum(axis=1)
    
    # Center the array at 0
    total = total - total.mean()
    
    # Normalize
    max_abs_value = max(abs(max(total)), abs(min(total)))
    total = total / max_abs_value
    
    # Ruta completa al escritorio (Desktop) 
    desktop_path = "/Users/Educacion/Desktop/" #Se puede guardar en la carpeta deseada, aquí se utilizó como ejemplo la carpeta escritorio
    
    # Nombre del archivo de salida y ruta completa
    output_audio_file = desktop_path + "RuidoRosa.wav"
    # Agregar generación de archivo de audio .wav
    sf.write("RuidoRosa.wav", total, 44100)
    # Guarda el nuevo array amplificado de numpy como archivo de audio   
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
    print(test_ruido_rosa(tiempo,100))
    
