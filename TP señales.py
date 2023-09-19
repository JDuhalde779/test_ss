#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 10:30:41 2023


@author: jduhalde
"""

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def ampli(onda,valor):
    """ Esta función evalua el array ingresado y analiza si al multiplicarlo por  
        el valor de amplificación generaría saturación en caso de que esto no sucede se amplifica el array
    Argumentos:
        onda(array): se debe ingresar un array de numpy 
        valor(float): Es el valor por el que se quiere amplificar el array ingresado
        
    Returns: 
        (array) El return es el nuevo array generado luego de ser amplificado
    
    """
    maximo = np.max(onda)
    if np.abs(maximo * valor) > 1:
        print("El valor ingresado de amplificacion es incorrecto")
    else:
        amplificado = valor * onda
        
    return amplificado    

#Realización de función seno para testeo  
# Parámetros de la función seno      
amplitud = 0.4  # Amplitud de la seno
frecuencia = 1000  # Frecuencia en Hz (puedes ajustarla)
duracion = 5.0  # Duración en segundos (puedes ajustarla)
frecuencia_muestreo = 44100  # Frecuencia de muestreo en Hz (puedes ajustarla)

# Crea un arreglo de tiempo
tiempo = np.linspace(0, duracion, int(duracion * frecuencia_muestreo), endpoint=False)

# Crea la función seno
seno = amplitud * np.sin(2 * np.pi * frecuencia * tiempo)


valor_de_ampli=2

def test_ampli(seno,valor_de_ampli):
    testeo = ampli(seno, valor_de_ampli)
    if np.abs(np.max(testeo)) > 1:
        return False
    else:
        return (True,testeo)

 # Llama a la función test_ampli y guarda el resultado en ploteo
[onda,ploteo] = test_ampli(seno, valor_de_ampli)
print(onda)
 
# Grafica la función seno
plt.figure(figsize=(10, 4))
plt.plot(tiempo, ploteo)
plt.title('Función Seno Generada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.xlim(0, 0.005)
plt.show()

# Ruta completa al escritorio (Desktop) 
desktop_path = "/Users//Desktop/" #Se puede guardar en la carpeta deseada, aquí se utilizó como ejemplo la carpeta escritorio

# Nombre del archivo de salida y ruta completa
output_audio_file = desktop_path + "audio_amplificado.wav"

# Guarda el nuevo array amplificado de numpy como archivo de audio
sf.write(output_audio_file, ploteo, frecuencia_muestreo)

print(f"El archivo {output_audio_file} se ha guardado en el escritorio.")






