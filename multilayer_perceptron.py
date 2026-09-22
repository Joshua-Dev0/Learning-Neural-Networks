import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "mnist_train.csv")
df = pl.read_csv(csv_path)

num_rows = df.height
pixel_matrix = df.drop("label").to_numpy()
pixel_matrix_norm = pixel_matrix / 255.0   #Normalize your pixel data

def initparams():
    low_bound = -0.1
    high_bound = 0.1

    #Initializing random data
    # Generate a uniform distribution of float numbers for hidden layer 1
    W1 = np.random.uniform(low=low_bound, high=high_bound, size=(64, 784))
    B1 = np.zeros((64, 1))

    # Generate a uniform distribution of float numbers for hidden layer 2
    W2 = np.random.uniform(low=low_bound, high=high_bound, size=(10, 64))
    B2 = np.zeros((10, 1))

    return W1, B1, W2, B2

def softmax(Z: np.ndarray):
    # Subtracting the max prevents numerical explosion (overflow errors)
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z)

def relu(Z: np.ndarray):
    # Compares every single number in Z against 0 and keeps the larger value
    return np.maximum(0, Z)

def relu_deriv(Z: np.ndarray):
    # (Z > 0) creates True/False. .astype(float) turns True->1.0 and False->0.0
    return (Z > 0).astype(float)

def one_hot(label: int) -> np.ndarray:
    # 1. Initialize a 10-row, 1-column array filled with 0.0
    array = np.zeros((10, 1))
    
    # 2. Set the index of the true digit to 1.0
    array[label, 0] = 1.0
    
    return array

def compute_cost(A2: np.ndarray, Y_one_hot: np.ndarray):
    # 1. Prevent log(0) errors which yield NaN (Not a Number) crashes
    epsilon = 1e-15
    # 1.0 - epsilon is the ceiling height, i.e 0.999999999
    # epsilon is the floor limit, i.2 0.0000000001
    # A2 is limited between these two limits
    A2_clipped = np.clip(A2, epsilon, 1.0 - epsilon)
    
    # 2. Apply the cross-entropy formula: -Σ (Y * log(A2))
    cost = -np.sum(Y_one_hot * np.log(A2_clipped))
    
    return cost

def accuracy_check(result: np.ndarray, onehot: np.ndarray):
    return np.sum(result == onehot) / onehot.shape[0]


def forward_propagation(X: np.ndarray, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):

    # Pass 1: Hidden Layer calculation (Outputs a 64x1 matrix)
    Z1 = (W1 @ X) + B1
    A1 = relu(Z1)  
    
    # Pass 2: Output Layer calculation (Outputs a 10x1 matrix)
    Z2 = (W2 @ A1) + B2  # (10, 64) @ (64, 1) + (10, 1) = (10, 1)
    A2 = softmax(Z2)     # Scales the 10 raw digits scores into probabilities
    
    return A2, Z2, A1, Z1

def backward_propagation(i, X: np.ndarray, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):

    # i for every row(every single image)
    A2, Z2, A1, Z1 = forward_propagation(X, pixel_matrix_norm, W1, B1, W2, B2)
    cost = compute_cost(A2, one_hot(df[i, 0]))
    

    return W1, B1, W2, B2

def main():
    W1, B1, W2, B2 = initparams()   #Initialize Parameters

    csv_num_row = 0 # Row 1 (image 1)
    data_row = pixel_matrix_norm[csv_num_row]  # get first row
    X = data_row.reshape(784, 1) # reshape row to column

    # for i in range(100):
    # W1, B1, W2, B2 = backward_propagation(i, X, W1, B1, W2, B2)
    A2, Z2, A1, Z1 = forward_propagation(X, W1, B1, W2, B2)

    print("Probabilities:")
    for i in range(10):
        print(f"{i}:", A2[i])

    #find the highest value
    digit = np.argmax(A2)
    print("Predicted digit:", digit)

    #actual number
    label = df[csv_num_row, 0]
    print("Labels:", label)

    Y_onehot = one_hot(df[i, 0])
    print("Cost: ", compute_cost(A2, Y_onehot))
    print("Prediction Accuracy: ", accuracy_check(A2, Y_onehot))

    # Display currently used image row
    imageGrid = data_row.reshape(28, 28) # 2. Reshape it back into a 28x28 matrix grid
    plt.imshow(imageGrid, cmap='gray') # 3. Plot the image using a grayscale color map ('gray')
    plt.axis('off') # 4. (Optional) Turn off the coordinate axis lines for a cleaner look
    plt.show() # 5. Display the window on your screen

main()