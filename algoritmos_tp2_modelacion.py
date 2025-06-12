# SE REQUIEREN ESTAS BIBLIOTECAS PARA QUE
# EL CODIGO FUNCIONE CORRECTAMENTE
import matplotlib.pyplot as plt
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
    # AQUI EN EL MAIN SE PUEDEN PROBAR LOS DISTINTOS ALGORITMOS,
    # VER SUS GRAFICOS, Y ESTIMAR ERRORES DE TRUNCAMIENTO

    # EL METODO A UTILIZAR RECIBE COMO PARAMETRO EL PASO DE CALCULO
    solucion_paper = solucion_analitica(0.025)
    solucion_euler = resolucion_euler(4)
    solucion_pc = resolucion_predictor_corrector(4)
    solucion_rk4  = resolucion_runge_kutta_4(4)

    # AGREGAR A ESTA LISTA DE SOLUCIONES, LAS SOLUCIONES A GRAFICAR
    # CON EL NOMBRE, PASO DE CALCULO, LA SOLUCION, EL COLOR Y TRUE O FALSE PARA GRAFICAR LOS PUNTOS
    soluciones = [
        ("Solucion Paper", 0.025, solucion_paper, "black", False),
        ("Solucion Euler", 4, solucion_euler, "red", True),
        ("Solucion P-C", 4, solucion_pc, "green", True),
        ("Solucion RK4", 4, solucion_rk4, "blue", True),
    ]
    
    # LA FUNCION DE CALCULAR ERRORES DE TRUNCAMIENTO RECIBE DOS SOLUCIONES 
    # PARA COMPARARLAS
    error_euler = calcular_errores_truncamiento(solucion_paper, solucion_euler)
    error_pc = calcular_errores_truncamiento(solucion_paper, solucion_pc)
    error_rk4 = calcular_errores_truncamiento(solucion_paper, solucion_rk4)


    # METER LOS ERRORES CALCULADOS EN ESTA LISTA, COMO UNA TUPLA CON
    # EL NOMBRE DEL METODO Y EL PASO, Y LUEGO EL ERROR
    errores = [("Err trunc Euler(dx=4)", error_euler),
                ("Err trunc P-C(dx=4)", error_pc),
                ("Err trunc RK4(dx=4)", error_rk4)
            ]
    

    graficar_soluciones(soluciones, errores)

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

main()