import numpy as np

def ReLu(x):
    return np.maximum(0, x)

def ReLuDerivative(x):
    return (x > 0).astype(int)

print(ReLuDerivative(np.array([-2, -1, 0, 1, 2])))

