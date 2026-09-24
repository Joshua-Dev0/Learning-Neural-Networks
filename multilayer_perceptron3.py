import os
import polars as pl
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path_train = os.path.join(SCRIPT_DIR, "mnist_train.csv")
csv_path_test = os.path.join(SCRIPT_DIR, "mnist_test.csv")
model_path = os.path.join(SCRIPT_DIR, "model.npz")

df_train = pl.read_csv(csv_path_train)
df_test = pl.read_csv(csv_path_test)

tr_label = df_train["label"].to_numpy()
ts_label = df_test["label"].to_numpy()

tr_rows = df_train.height
tr_columns = df_train.width - 1

ts_rows = df_test.height
ts_columns = df_test.width - 1

tr_data = cp.asarray(df_train.drop("label").to_numpy())
ts_data = cp.asarray(df_test.drop("label").to_numpy())

tr_norm = tr_data / 255.0
ts_norm = ts_data / 255.0


class Layer:
  def __init__(self, row, col):
    self.weights = cp.random.uniform(low=-0.1, high=0.1, size=(row, col))
    self.bias = cp.zeros((row, 1))

def initparams():
  hidden_layers = [Layer(64, 784) for _ in range(3)]
  output_layer = Layer(10, 64)
  
  return hidden_layers, output_layer

def onehot(label: int):
  label = int(label)
  array = cp.zeros((10, 1))
  array[label, 0] = 1.0

  return array

def softmax(Z):
  exp_Z = cp.exp(Z - cp.max(Z))

  return exp_Z / cp.sum(exp_Z)

def relu(Z):
  return cp.maximum(0, Z)

def relu_deriv(Z):

  return (Z > 0).astype(cp.float32)

def accuracy(result, onehot):

  prediction = cp.argmax(result)
  actual = cp.argmax(onehot)

  return bool(prediction == actual)



def forward_propagation(X, Layers):
  A = X
  As = []
  Zs = []
  
  for layer in layers:
    Z = layer.weights @ A + layer.bias
    A = relu(Z)
    
    Zs.append(Z)
    As.append(A)

  return As, Zs 

def backward_propagation(Y_onehot, X, As, Zs, Layers):

  # --------------------------------------------------------
  # Output layer
  # --------------------------------------------------------

  dZ_last = A2 - Y_onehot

  dW_last = dZ2 @ A1.T

  dB_last = dZ2
  
  for layer in layers:


  # --------------------------------------------------------
  # Hidden layer
  # --------------------------------------------------------

    dA1 = W2.T @ dZ2

    dZ1 = dA1 * relu_deriv(Z1)

    dW1 = dZ1 @ X.T

    dB1 = dZ1


  return dW2, dB2, dW1, dB1

def update_params(alpha, W1, B1, W2, B2, dW2, dB2, dW1, dB1):

  W1 = W1 - (alpha * dW1)
  B1 = B1 - (alpha * dB1)
  W2 = W2 - (alpha * dW2)
  B2 = B2 - (alpha * dB2)

  return W1, B1, W2, B2

def gradient_descent(iterations, alpha, W1, B1, W2, B2):

  print("Progress:")

  for epoch in tqdm(range(iterations)):

    for i in range(tr_rows):

      X = tr_norm[i].reshape(784, 1)

      A2, Z2, A1, Z1 = forward_propagation(X, W1, B1, W2, B2)

      Y_onehot = onehot(tr_label[i])

      backward_propagation(Y_onehot, X, A2, Z2, A1, Z1, W1, W2)

      W1, B1, W2, B2 = update_params(alpha, W1, B1, W2, B2, dW2, dB2, dW1, dB1)

  return W1, B1, W2, B2

def main():
  hidden_layers, output_layer = initparams()
  
  cont = 1

  print("Training Label:", tr_label[0])


  while cont == 1:

    print("Options ====================")

    print("- R for Train")
    print("- T for Test")
    print("- L for Load")
    print("- S for Save")

    option = input("Select an option: ")

    if option == "R":

      iterations = int(
        input("Iterations: ")
      )

      alpha = 0.01

      W1, B1, W2, B2 = gradient_descent(iterations, alpha, W1, B1, W2, B2)

    elif option == "T":

      index = int( input("Select Index: ") )

      X = ts_norm[index].reshape(784, 1)

      A, Z = forward_propagation(X, W1, B1, W2, B2)

      Y_onehot = onehot(ts_label[index])

      print(
        "Prediction Accuracy:",
        accuracy(A2, Y_onehot)
      )

      print("Probabilities:")

      for i in range(10):
        print(
          f"{i}: {A2[i].item()}"
        )

      digit = int(cp.argmax(A2).item())

      print(
        "Predicted digit:",
        digit
      )

      label = int(ts_label[index])

      print(
        "Label:",
        label
      )

      imageGrid = ts_norm[index].reshape(28, 28)

      imageGrid_cpu = imageGrid.get()

      plt.imshow(
        imageGrid_cpu,
        cmap="gray"
      )

      plt.axis("off")

      plt.show()

    elif option == "L":

      model = cp.load(model_path)
      W1 = model["W1"]
      B1 = model["B1"]
      W2 = model["W2"]
      B2 = model["B2"]

    elif option == "S":
      
      cp.savez(
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
