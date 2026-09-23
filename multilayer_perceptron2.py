import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path_train = os.path.join(SCRIPT_DIR, "mnist_train.csv")
csv_path_test = os.path.join(SCRIPT_DIR, "mnist_test.csv")
df_train = pl.read_csv(csv_path_train)
df_test = pl.read_csv(csv_path_test)

tr_label = df_train["label"].to_numpy()
ts_label = df_test["label"].to_numpy()

tr_rows = df_train.height
tr_columns = df_train.width - 1 # -1 to exclude the labels

ts_rows = df_test.height
ts_columns = df_test.width - 1 # -1 to exclude the labels

tr_data = df_train.drop("label").to_numpy()
ts_data = df_test.drop("label").to_numpy()

tr_norm = tr_data / 255.0   #Normalize your pixel data, squishing numbers to be between 0 and 1
ts_norm = ts_data / 255.0   #255.0 is the dataset's maximum value

def initparams():
  low_bound = -0.1
  high_bound = 0.1

  W1 = np.random.uniform(low=low_bound, high=high_bound, size=(64, 784))
  B1 = np.zeros((64, 1))

  W2 = np.random.uniform(low=low_bound, high=high_bound, size=(10, 64))
  B2 = np.zeros((10, 1))

  return W1, B1, W2, B2

def onehot(label: int) -> np.ndarray:
  label = int(label)
  array = np.zeros((10, 1))
  array[label, 0] = 1.0
  return array

def softmax(Z: np.ndarray):
  exp_Z = np.exp(Z - np.max(Z))
  return exp_Z / np.sum(exp_Z)

def relu(Z: np.ndarray):
  return np.maximum(0, Z)

def relu_deriv(Z: np.ndarray):
  # (Z > 0) creates True/False. .astype(float) turns True->1.0 and False->0.0
  return (Z > 0).astype(float)

def accuracy(result: np.ndarray, onehot: np.ndarray):
  prediction = np.argmax(result)
  actual = np.argmax(onehot)

  return prediction == actual



def forward_propagation(X: np.ndarray, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):
  # Pass 1: Hidden Layer calculation (Outputs a 64x1 matrix)
  Z1 = (W1 @ X) + B1  # ([64x784] * [784x1]) + [64x1] = [64x1]
  A1 = relu(Z1)  
  
  # Pass 2: Output Layer calculation (Outputs a 10x1 matrix)
  Z2 = (W2 @ A1) + B2  # ([10x64] * [64, 1]) + [10x1] = [10x1]
  A2 = softmax(Z2)     # Scales the 10 raw digits scores into probabilities
  
  return A2, Z2, A1, Z1

def backward_propagation(Y_onehot: np.ndarray, X: np.ndarray, A2: np.ndarray, Z2: np.ndarray, A1: np.ndarray, Z1: np.ndarray, W1: np.ndarray, W2: np.ndarray):
  
  dZ2 = A2 - Y_onehot    #Cost + softmax function derivative
  dW2 = dZ2 @ A1.T    # for each image trained, dW2 = (1/m) * (dZ2 @ A1.T) for batch size m
  dB2 = dZ2
  
  dA1 = W2.T @ dZ2
  dZ1 = dA1 * relu_deriv(Z1)
  dW1 = dZ1 @ X.T
  dB1 = dZ1
  
  return dW2, dB2, dW1, dB1

def update_params(alpha, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray, dW2: np.ndarray, dB2: np.ndarray, dW1: np.ndarray, dB1: np.ndarray):
  
  W1 = W1 - (alpha * dW1)
  B1 = B1 - (alpha * dB1)
  W2 = W2 - (alpha * dW2)
  B2 = B2 - (alpha * dB2)
  
  return W1, B1, W2, B2

def gradient_descent(iterations, alpha, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):
  print("Progress: ")
  for epoch in tqdm(range(iterations)):
    
    for i in range(tr_rows):
      X = tr_norm[i].reshape(784, 1)
      A2, Z2, A1, Z1 = forward_propagation(X, W1, B1, W2, B2)
      
      Y_onehot = onehot(tr_label[i])
      # print("Prediction Accuracy: ", accuracy(A2, Y_onehot))
      
      #debug
      # print("X: ", X.shape)
      # print("W1:", W1.shape)
      # print("Z1:", Z1.shape)
      # print("A1:", A1.shape)
      # print("W2:", W2.shape)
      # print("Z2:", Z2.shape)
      # print("A2:", A2.shape)
      # print("Y: ", Y_onehot.shape)
      
      dW2, dB2, dW1, dB1 = backward_propagation(Y_onehot, X, A2, Z2, A1, Z1, W1, W2)
      W1, B1, W2, B2 = update_params(alpha, W1, B1, W2, B2, dW2, dB2, dW1, dB1)
      
  return W1, B1, W2, B2

def main():
  W1, B1, W2, B2 = initparams()
  cont = 1
  
  # Debug
  print("Training Label: ", tr_label[0])
  
  while cont == 1:
    print("Options ====================")
    print("- R for Train")
    print("- T for Test")
    print("- L for Load")
    print("- S for Save")
    option = input("Select an option: ")
    
    if(option == "R"):
      iterations = int(input("Iterations: "))
      alpha = 0.01
      W1, B1, W2, B2 = gradient_descent(iterations, alpha, W1, B1, W2, B2)
      
    elif(option == "T"):
      index = int(input("Select Index: "))
      X = ts_norm[index].reshape(784, 1)
      A2, Z2, A1, Z1 = forward_propagation(X, W1, B1, W2, B2)
      
      Y_onehot = onehot(ts_label[index])
      print("Prediction Accuracy: ", accuracy(A2, Y_onehot))
      
      #debug
      # print("X: ", X.shape)
      # print("W1:", W1.shape)
      # print("Z1:", Z1.shape)
      # print("A1:", A1.shape)
      # print("W2:", W2.shape)
      # print("Z2:", Z2.shape)
      # print("A2:", A2.shape)
      # print("Y: ", Y_onehot.shape)
      
      #Results
      print("Probabilities:")
      for i in range(10):
        print("{i}:", A2[i])

      #find the highest value
      digit = np.argmax(A2)
      print("Predicted digit:", digit)

      #actual number
      label = ts_label[index]
      print("Label:", label)

      # Display currently used image row
      imageGrid = ts_norm[index].reshape(28, 28) # 2. Reshape it back into a 28x28 matrix grid
      plt.imshow(imageGrid, cmap='gray') # 3. Plot the image using a grayscale color map ('gray')
      plt.axis('off') # 4. (Optional) Turn off the coordinate axis lines for a cleaner look
      plt.show() # 5. Display the window on your screen
      
    elif(option == "L"):
      model = np.load("model.npz")
      W1 = model["W1"]
      B1 = model["B1"]
      W2 = model["W2"]
      B2 = model["B2"]
  
    elif(option == "S"):
      np.savez(
        "model.npz",
        W1=W1,
        B1=B1,
        W2=W2,
        B2=B2
      )
      
    else:
      print("Invalid Input")
      
    print("Continue ====================")
    print("- 1 for Continue")
    print("- 0 for False")
    cont = int(input("Select an option: "))

main()