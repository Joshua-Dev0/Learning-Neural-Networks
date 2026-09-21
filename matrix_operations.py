import numpy as np

# Two 2×2 matrices
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

# Matrix addition
C = A + B

# Matrix multiplication
D = A @ B

print("A:")
print(A)

print("\nB:")
print(B)

print("\nA + B:")
print(C)

print("\nA × B:")
print(D)