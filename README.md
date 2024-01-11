# OptimizacionAPI
 API de metodo simplex con FastAPI para poder utilizar otras dimensiones cambiar:


SIZE_MATRIX_X = 1
SIZE_MATRIX_Y = 6

# USO
uvicorn main:app --reload

# ¿Como mandas requests?
## Requests de ejemplos: 

### 1.- 
curl -X POST "http://127.0.0.1:8000/solve" -H "Content-Type: application/json" -d '{"A": [[1, -1, 20, 10, -1], [3, 2, 81, 4, 2], [0, 1, -1, 1, 1]], "b": [4, 5, 2], "c": [1, -2, 10, 1, 30]}'

### 2.-
curl -X POST "http://127.0.0.1:8000/solve" -H "Content-Type: application/json" -d '{"A": [[1, -1, 2, 0, -1], [3, 2, 1, 4, 2], [0, 1, -1, 1, 1]], "b": [4, 5, 2], "c": [1, -2, 0, 1, 3]}'

