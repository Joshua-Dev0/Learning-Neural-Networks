import os
import polars as pl
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "mnist_train.csv")

df = pl.read_csv(csv_path)

num_rows = df.height
pixel_matrix = df.drop("label").to_numpy()
pixel_matrix_norm = pixel_matrix / 255.0   #Normalize your pixel data

low_bound = -0.1
high_bound = 0.1

#Initializing random data
# Generate a uniform distribution of float numbers for hidden layer 1
W1 = np.random.uniform(low=low_bound, high=high_bound, size=(64, 784))
B1 = np.random.uniform(low=low_bound, high=high_bound, size=(64, 1))

# Generate a uniform distribution of float numbers for hidden layer 2
W2 = np.random.uniform(low=low_bound, high=high_bound, size=(10, 64))
B2 = np.random.uniform(low=low_bound, high=high_bound, size=(10, 1))


def softmax(Z: np.ndarray):
    # Subtracting the max prevents numerical explosion (overflow errors)
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z)

def relu(Z: np.ndarray):
    # Compares every single number in Z against 0 and keeps the larger value
    return np.maximum(0, Z)

def reluderiv(Z: np.ndarray):
    # (Z > 0) creates True/False. .astype(float) turns True->1.0 and False->0.0
    return (Z > 0).astype(float)

# input: np.ndarray, bias: np.ndarray, weights: np.ndarray
def forward_propagation(csv_row, rows_of_pixels: np.ndarray, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):

    #extract specified row from pixels
    first_row_pixels = rows_of_pixels[csv_row]

    #reshape flat 1D array to a single column with 784 rows
    X = first_row_pixels.reshape(784, 1)

    # Pass 1: Hidden Layer calculation (Outputs a 64x1 matrix)
    Z1 = (W1 @ X) + B1
    A1 = relu(Z1)  
    
    # Pass 2: Output Layer calculation (Outputs a 10x1 matrix)
    Z2 = (W2 @ A1) + B2  # (10, 64) @ (64, 1) + (10, 1) = (10, 1)
    A2 = softmax(Z2)     # Scales the 10 raw digits scores into probabilities
    
    return A2

def main():
    csv_num_row = 0
    result = forward_propagation(csv_num_row, pixel_matrix_norm, W1, B1, W2, B2)

    print("Probabilities:")
    for i in range(10):
        print(f"{i}:", result[i])

    #find the highest value
    digit = np.argmax(result)
    print("Predicted digit:", digit)
    #actual number
    label = df[csv_num_row, 0]
    print("Labels:", label)

main()