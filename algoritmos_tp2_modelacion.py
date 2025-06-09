import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import math

# ESTOS VALORES ESTAN FIJOS
H=0.8
h=1
k=0.2
r=0.35
ecgs = r**2*h**(5/2)
ecg_inicial = H**2*h**(1/2)
limite_tn=24
############################

def main():
    solucion_paper = solucion_analitica(0.025)
    rungekutta8 = resolucion_runge_kutta_4(8)
    rungekutta9 = resolucion_runge_kutta_4(9)
    rungekutta10 = resolucion_runge_kutta_4(10)
    rungekutta11 = resolucion_runge_kutta_4(11)
    rungekutta12 = resolucion_runge_kutta_4(12)
    rungekutta13 = resolucion_runge_kutta_4(13)
    rungekutta14 = resolucion_runge_kutta_4(14)

    soluciones = [
        ("Solucion Paper", 0.025, solucion_paper, "black", False),
        ("Runge-Kutta 4", 8, rungekutta8, "red", True),
        ("Runge-Kutta 4", 9, rungekutta9, "green", True),
        ("Runge-Kutta 4", 10, rungekutta10, "blue", True),
        ("Runge-Kutta 4", 11, rungekutta11, "yellow", True),
        ("Runge-Kutta 4", 12, rungekutta12, "purple", True),
        ("Runge-Kutta 4", 13, rungekutta13, "violet", True),
        ("Runge-Kutta 4", 14, rungekutta14, "orange", True),

    ]
    errorrungekutta8 = calcular_errores_truncamiento(solucion_paper, rungekutta8)
    errorrungekutta9 = calcular_errores_truncamiento(solucion_paper, rungekutta9)
    errorrungekutta10 = calcular_errores_truncamiento(solucion_paper, rungekutta10)
    errorrungekutta11 = calcular_errores_truncamiento(solucion_paper, rungekutta11)
    errorrungekutta12 = calcular_errores_truncamiento(solucion_paper, rungekutta12)
    errorrungekutta13 = calcular_errores_truncamiento(solucion_paper, rungekutta13)
    errorrungekutta14 = calcular_errores_truncamiento(solucion_paper, rungekutta14)

    graficar_soluciones(soluciones, 
                        [
                        ("Err trunc RK4(dx=8)", errorrungekutta8),
                        ("Err trunc RK4(dx=9)", errorrungekutta9),
                        ("Err trunc RK4(dx=10)", errorrungekutta10),
                        ("Err trunc RK4(dx=11)", errorrungekutta11),
                        ("Err trunc RK4(dx=12)", errorrungekutta12),
                        ("Err trunc RK4(dx=13)", errorrungekutta13),
                        ("Err trunc RK4(dx=14)", errorrungekutta14),
                        ]
                        )

def graficar_soluciones(lista_soluciones, errores):
    """
    Cada elemento de lista_de_soluciones debe ser una tupla:
    (nombre_del_metodo, dx_utilizado, lista_de_tuplas_(x_i, H_i), color, se_grafican_puntos=bool)
    """
    plt.figure(figsize=(8, 5))

    for nombre, dx, coordenadas, color, graficar_puntos in lista_soluciones:
        valores_x = [x for x, H in coordenadas]
        valores_H = [H for x, H in coordenadas]
        if graficar_puntos:
            plt.plot(valores_x, valores_H, marker='o', linestyle='-', color=color, label=f"{nombre} (dx={dx})")
        else:
            plt.plot(valores_x, valores_H, linestyle='-', color=color, label=f"{nombre} (dx={dx})")

    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlabel('x (metros)')
    plt.ylabel('Altura H (metros)')
    plt.title('Evolucion de la altura de la ola')
    plt.legend()

    # Aca se muestran los errores
    texto_errores = "\n".join([f"{etiqueta}: {valor:.6f}" for etiqueta, valor in errores])
    plt.text(0.65, 0.05, texto_errores,
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(np.arange(0, 25, 4))
    plt.show()

def f(ecg):
    return (-k / h) * (ecg - ecgs)

def calcular_H_actual(ecg):
    return (ecg / h**(1/2))**(1/2)

def arange_manual(inicio, fin, paso):
    valores = []
    actual = inicio
    while actual <= fin:
        valores.append(round(actual, 10))
        actual += paso
    return valores

def solucion_analitica(dx):
    x_vals = arange_manual(0, limite_tn, dx)
    H_vals = []
    for x in x_vals:
        ecg_actual = ecgs + (ecg_inicial - ecgs) * math.exp(-k * x / h)
        H_actual = calcular_H_actual(ecg_actual)
        H_vals.append(H_actual)

    return list(zip(x_vals, H_vals))

def resolucion_euler(dx):
    ecg_actual = ecg_inicial
    H_actual = H
    tn = 0
    tuplas_tn_H = [(tn, H_actual)]

    while tn < limite_tn:
        ecg_actual = ecg_actual + dx*f(ecg_actual)
        H_actual = calcular_H_actual(ecg_actual)
        tn += dx
        tuplas_tn_H.append((tn, H_actual))
    
    return tuplas_tn_H

def resolucion_predictor_corrector(dx):
    ecg_actual = ecg_inicial
    H_actual = H
    tn = 0
    tuplas_tn_H = [(tn, H_actual)]

    while tn < limite_tn:
        ecg_sig_medio = ecg_actual + (dx/2)*f(ecg_actual)
        ecg_actual = ecg_actual + dx*f(ecg_sig_medio)
        H_actual = (ecg_actual / h**(1/2))**(1/2)

        tn += dx
        tuplas_tn_H.append((tn, H_actual))

    return tuplas_tn_H

def resolucion_runge_kutta_4(dx):
    ecg_actual = ecg_inicial
    H_actual = H
    tn = 0
    tuplas_tn_H = [(tn, H_actual)]
    
    while tn < limite_tn:
        q1 = dx * f(ecg_actual)
        q2 = dx * f(ecg_actual + (1/2)*q1)
        q3 = dx * f(ecg_actual + (1/2)*q2)
        q4 = dx * f(ecg_actual + q3)

        ecg_siguiente = ecg_actual + (1/6)*(q1 + 2*q2 + 2*q3 + q4)
        ecg_actual = ecg_siguiente
        H_actual = (ecg_actual / h**(1/2))**(1/2)

        tn += dx
        tuplas_tn_H.append((tn, H_actual))

    return tuplas_tn_H

def calcular_errores_truncamiento(puntos_solucion_exacta, puntos_solucion_aproximada):
    """Se suma la distancia entre todos los puntos en comun de ambas soluciones
    para aproximar el error de truncamiento"""
    error_truncamiento = 0
    for punto1 in puntos_solucion_exacta:
        for punto2 in puntos_solucion_aproximada:
            if punto1[0] == punto2[0]:
                error_truncamiento += abs(punto1[1] - punto2[1])
    return error_truncamiento

def mostrar_ultimas_iteraciones(tabla_alturas_iteraciones):
    ultimos_diez = tabla_alturas_iteraciones[-10:]
    tabla_formateada = tabulate(
        ultimos_diez,
        headers=["Iteración", "ECg"],
        tablefmt="fancy_grid",
        floatfmt=".6f"
    )
    print(tabla_formateada)

main()