import numpy as np
from pandas import DataFrame as pdData
import math
from fastapi import FastAPI, Query

app = FastAPI()

# Constants for matrix dimensions
SIZE_MATRIX_X = 1
SIZE_MATRIX_Y = 6

def generate_tabinitial(A, b, c):
    """
    Generates the initial tableau for the simplex method.

    Args:
        A: Constraint matrix
        b: Right-hand side vector
        c: Objective function coefficients

    Returns:
        A tuple containing:
        - A range object for the artificial variables
        - The initial tableau as a NumPy array
    """
    [m, n] = A.shape
    if m != b.size or n != c.size:
        raise ValueError("Invalid input dimensions")

    b = b.reshape((m, 1))  # Reshape b to be a column vector
    c = c.reshape((1, n))  # Reshape c to have the same number of columns as A

    result = np.column_stack((A, b))
    result = np.append(result, np.column_stack((c, 0)), axis=0)
    
    return range(n, n + m), result


def generate_tabinitial_withID(A, b, c):
    """
    Generates the initial tableau with identity matrix for artificial variables.

    Args:
        A: Constraint matrix
        b: Right-hand side vector
        c: Objective function coefficients

    Returns:
        A tuple containing:
        - A range object for the artificial variables
        - The initial tableau with identity matrix as a NumPy array
    """
    m, n                = A.shape
    rng, tabinitial     = generate_tabinitial(A, b, c)
    identity            = np.vstack((np.identity(m), np.zeros(m)))
    return rng, np.concatenate((tabinitial, identity), axis=1)

def positive(v):
    """
    Checks if all elements of a vector are non-negative.

    Args:
        v: The vector to check

    Returns:
        A tuple containing:
        - True if all elements are non-negative, False otherwise
        - The minimum value in the vector
        - The indices of the minimum value
    """
    return all(v >= 0), np.amin(v), np.where(v == np.amin(v))


def init(tab, A):
    """
    Initializes variables for the simplex algorithm.

    Args:
        tab: The tableau
        A: Constraint matrix

    Returns:
        A tuple containing:
        - The current optimal value
        - The current b vector
        - The current c vector
        - The current A matrix
    """
    m, n    = A.shape

    opt         = -tab[m, n] # Negative of the bottom-right element is the current optimal value
    tab_b       = tab[:m, n] # Extract the b vector
    tab_c       = np.concatenate((tab[m , 0:n] ,tab[m , n + 1:]))  # Combine c vector parts

    tab_A       = np.hstack((tab[0:m ,0:n ] ,tab[ 0:m , n + 1 :])) # Combine A matrix parts

    return opt, tab_b, tab_c, tab_A

def index_smallest_pos(v):
    """
    Finds the index of the smallest positive element in a vector.

    Args:
        v: The vector to search

    Returns:
        The index of the smallest positive element, or -1 if none exist
    """
    return np.where(v > 0, v, np.inf).argmin()

def rapportmin(a, b, m):
    """
    Calculates the minimum ratio of corresponding elements in two vectors.

    Args:
        a: The first vector
        b: The second vector
        m: Number of elements to consider

    Returns:
        The index of the minimum ratio
    """
    out = []
    for i in range(0, m-1):
        if b[i] != 0:
            out.append(a[i] / b[i])
            
    return index_smallest_pos(np.array(out))

def resolution(tab, A, c):
    """
    Resolves the simplex method.
    
    Args:
        tab: The tableau
        A: Constraint matrix
        c: Objective function coefficients
    
    Returns:
        A tuple containing:
        - The optimal value
        - The optimal solution
    """
    opt, tab_b, tab_c, tab_A    = init(tab, A)
    m, n                        = tab.shape
    sign, minimum, index_min    = positive(tab_c)
    
    if (index_min[0] > c.size).all():
        index_min = list(index_min)
        index_min[0] += 1
        index_min = tuple(index_min)
    
    tab = tab.astype(np.float32)

    if sign:
        return tab_b, opt
    else:
        if all(tab[:,index_min] <= 0):
            print()
            raise ValueError("Error: the problem is unbounded")
        else:
            A_s                 = tab[:A.shape[0],index_min]

            index_pivot         = rapportmin(tab_b, A_s, m)

            ligne_pivot         = tab[index_pivot]
            colonne_pivot       = tab[:,index_min]

            pivot               = tab[index_pivot,index_min]
            
            tab[index_pivot]    = ligne_pivot / float(pivot)
            
            for i in range(0, len(tab)):
                if not np.array_equal(tab[i], tab[index_pivot]):
                    tab[i] = tab[i] - tab[index_pivot] * tab[i, index_min]

            print("\n",pdData(tab))

            return resolution(tab, A, c)


@app.post("/solve")
async def solve(data: dict):
    A = np.array(data.get('A', []))
    b = np.array(data.get('b', []))
    c = np.array(data.get('c', []))

    # Generate initial tableau
    range_tab, tab_initial = generate_tabinitial_withID(A, b, c)

    # Solve using the resolution function
    x, y = resolution(tab_initial, A, c)

    # Convert NumPy arrays to Python lists for JSON serialization
    x = x.tolist()
    y = float(y)  # Convert NumPy float32 to native Python float

    # Return the solution
    return {"optimal_solution": abs(y), "x": x}
