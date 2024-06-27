import numpy as np
import matplotlib.pyplot as plt

def ajuste_polinomico(x, y, grado):
    """
    Realiza un ajuste polinómico de grado especificado a los datos dados (x, y).

    Parameters:
    x : array-like
        Datos de entrada en el eje x.
    y : array-like
        Datos de entrada en el eje y.
    grado : int
        Grado del polinomio de ajuste.

    Returns:
    coeficientes : ndarray
        Coeficientes del polinomio de mejor ajuste.
    """
    # Convertir los arreglos a arrays de numpy
    x = np.array(x)
    y = np.array(y)
    
    # Ajuste polinómico de grado especificado
    coeficientes = np.polyfit(x, y, grado)
    
    return coeficientes

def evaluar_polinomio(coeficientes, x):
    """
    Evalúa el polinomio definido por los coeficientes dados en los puntos x.

    Parameters:
    coeficientes : array-like
        Coeficientes del polinomio.
    x : array-like
        Puntos donde se evalúa el polinomio.

    Returns:
    y : ndarray
        Valores del polinomio evaluado en los puntos x.
    """
    # Evaluar el polinomio en los puntos x
    y = np.polyval(coeficientes, x)
    return y

def calcular_error(x, y, coeficientes):
    """
    Calcula el error cuadrático entre los datos dados y la aproximación polinómica.

    Parameters:
    x : array-like
        Datos de entrada en el eje x.
    y : array-like
        Datos de entrada en el eje y.
    coeficientes : array-like
        Coeficientes del polinomio de ajuste.

    Returns:
    error : float
        Error cuadrático entre los datos reales y la aproximación polinómica.
    """
    # Evaluar el polinomio en los puntos x
    y_aproximado = evaluar_polinomio(coeficientes, x)
    
    # Calcular el error cuadrático
    error = np.sum((y_aproximado - y)**2)
    
    return error

# Ejemplo de uso
x = [4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3, 6.8, 7.1]
y = [102.56, 130.11, 113.18, 142.05, 167.53, 195.14, 224.87, 256.73, 299.5, 326.72]
grado = 1  # Cambia el grado del polinomio aquí

# Obtener los coeficientes del polinomio de mejor ajuste
coeficientes = ajuste_polinomico(x, y, grado)
print(f"Coeficientes del polinomio de grado {grado}: {coeficientes}")

# Calcular el error cuadrático
error = calcular_error(x, y, coeficientes)
print(f"Error cuadrático total: {error}")

# Generar valores de y ajustados para la gráfica
x_fit = np.linspace(min(x), max(x), 100)
y_fit = evaluar_polinomio(coeficientes, x_fit)

# Graficar los puntos y la curva de mejor ajuste
plt.scatter(x, y, color='blue', label='Datos')
plt.plot(x_fit, y_fit, color='red', label=f'Polinomio de grado {grado}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Ajuste polinómico de grado {grado}')
plt.grid(True)
plt.show()
