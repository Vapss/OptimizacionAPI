from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import linprog
from typing import List
from fastapi.responses import JSONResponse

app = FastAPI()

class IngredientData(BaseModel):
    costos: List[float]
    coeficientes_utilidad: List[float]
    cantidades_disponibles: List[float]

def generar_restricciones(cantidades_disponibles):
    """
    Genera las restricciones del problema de programación lineal.

    Args:
        cantidades_disponibles: Las cantidades disponibles de cada ingrediente.

    Returns:
        Una lista de restricciones.
    """
    restricciones = []
    for i in range(len(cantidades_disponibles)):
        restricciones.append([1] * (i + 1) + [0] * (len(cantidades_disponibles) - i - 1))
    return restricciones

def calcular_beneficio_maximo(coeficientes_costo, cantidades_optimas):
    """Calcula el beneficio máximo.

    Args:
        coeficientes_costo: Los coeficientes de costo de cada ingrediente.
        cantidades_optimas: Las cantidades óptimas de cada ingrediente.

    Returns:
        El beneficio máximo.
    """
    beneficio_total = sum(coef * cantidad for coef, cantidad in zip(coeficientes_costo, cantidades_optimas))
    return beneficio_total

@app.post("/optimizar/")
async def optimizar_ingredientes(ingredient_data: IngredientData):
    try:
        # Obtener datos de la solicitud
        costos = ingredient_data.costos
        coeficientes_utilidad = [-x for x in ingredient_data.coeficientes_utilidad]  # Se invierten los coeficientes de utilidad
        cantidades_disponibles = ingredient_data.cantidades_disponibles

        # Definir restricciones
        restricciones = generar_restricciones(cantidades_disponibles)

        # Definir límites de variables de decisión
        variables_decision = [(0, None)] * len(costos)

        # Resolver el problema de programación lineal
        resultado = linprog(c=costos, A_ub=restricciones, b_ub=cantidades_disponibles, bounds=variables_decision)

        # Preparar resultados
        resultados = {
            "cantidades_optimas": resultado.x.tolist(),
            "beneficio_maximo": calcular_beneficio_maximo(costos, resultado.x),
        }
        print(resultado)
        return JSONResponse(content=resultados)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
