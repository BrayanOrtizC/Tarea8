import numpy as np
import matplotlib.pyplot as plt

def ajuste_exponencial(x, y):
    """
    Realiza un ajuste exponencial a los datos dados (x, y) utilizando el método de mínimos cuadrados.

    Parameters:
    x : array-like
        Datos de entrada en el eje x.
    y : array-like
        Datos de entrada en el eje y.

    Returns:
    a : float
        Coeficiente 'a' de la función exponencial y = be^(ax).
    b : float
        Coeficiente 'b' de la función exponencial y = be^(ax).
    """
    # Transformar los datos de y aplicando logaritmo natural
    y_log = np.log(y)
    
    # Ajuste lineal a los datos transformados
    coeficientes = np.polyfit(x, y_log, 1)
    
    # Extraer los coeficientes 'a' y 'b' de la función exponencial
    a = coeficientes[0]
    b = np.exp(coeficientes[1])
    
    return a, b

def evaluar_exponencial(a, b, x):
    """
    Evalúa la función exponencial definida por los coeficientes dados en los puntos x.

    Parameters:
    a : float
        Coeficiente 'a' de la función exponencial y = be^(ax).
    b : float
        Coeficiente 'b' de la función exponencial y = be^(ax).
    x : array-like
        Puntos donde se evalúa la función exponencial.

    Returns:
    y : ndarray
        Valores de la función exponencial evaluada en los puntos x.
    """
    # Evaluar la función exponencial en los puntos x
    y = b * np.exp(a * x)
    return y

def calcular_error_exponencial(x, y, a, b):
    """
    Calcula el error cuadrático entre los datos dados y la aproximación exponencial.

    Parameters:
    x : array-like
        Datos de entrada en el eje x.
    y : array-like
        Datos de entrada en el eje y.
    a : float
        Coeficiente 'a' de la función exponencial y = be^(ax).
    b : float
        Coeficiente 'b' de la función exponencial y = be^(ax).

    Returns:
    error : float
        Error cuadrático entre los datos reales y la aproximación exponencial.
    """
    # Evaluar la función exponencial en los puntos x
    y_aproximado = evaluar_exponencial(a, b, x)
    
    # Calcular el error cuadrático
    error = np.sum((y_aproximado - y)**2)
    
    return error

# Ejemplo de uso
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3, 6.8, 7.1])
y = np.array([102.56, 130.11, 113.18, 142.05, 167.53, 195.14, 224.87, 256.73, 299.5, 326.72])

# Obtener los coeficientes de la función exponencial
a, b = ajuste_exponencial(x, y)
print(f"Coeficiente 'a': {a}")
print(f"Coeficiente 'b': {b}")

# Calcular el error cuadrático
error = calcular_error_exponencial(x, y, a, b)
print(f"Error cuadrático total: {error}")

# Generar valores de y ajustados para la gráfica
x_fit = np.linspace(min(x), max(x), 100)
y_fit = evaluar_exponencial(a, b, x_fit)

# Graficar los puntos y la curva de mejor ajuste
plt.scatter(x, y, color='blue', label='Datos')
plt.plot(x_fit, y_fit, color='red', label='Ajuste exponencial')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Ajuste exponencial de los datos')
plt.grid(True)
plt.show()
