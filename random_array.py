import numpy as np

# Create a single-column array with 785 rows of random decimals between 0 and 1
# random_array = np.random.rand(785, 1)

# # Verify the shape of the array
# print("Array Shape:", random_array.shape) 
# # Output: Array Shape: (785, 1)

# # View the first few rows
# print(random_array[:5])

# Define your constraints
low_bound = -10
high_bound = 10
matrix_size = (64, 784)

# Generate a uniform distribution of float numbers between -10.0 and 10.0
W1 = np.random.uniform(low=low_bound, high=high_bound, size=matrix_size)

# Verify the matrix layout
print("Matrix Shape:", W1.shape)
print("Minimum value in matrix:", W1.min())
print("Maximum value in matrix:", W1.max())
