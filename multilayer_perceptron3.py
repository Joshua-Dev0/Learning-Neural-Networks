import os
import polars as pl
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path_train = os.path.join(SCRIPT_DIR, "mnist_train.csv")
csv_path_test = os.path.join(SCRIPT_DIR, "mnist_test.csv")

df_train = pl.read_csv(csv_path_train)
df_test = pl.read_csv(csv_path_test)

# These can remain NumPy because they come from Polars.
tr_label = df_train["label"].to_numpy()
ts_label = df_test["label"].to_numpy()

tr_rows = df_train.height
ts_rows = df_test.height

tr_data = df_train.drop("label").to_numpy()
ts_data = df_test.drop("label").to_numpy()

# Move training/test data to GPU
tr_norm = cp.asarray(tr_data / 255.0)
ts_norm = cp.asarray(ts_data / 255.0)


def initparams():
  low_bound = -0.1
  high_bound = 0.1

  # Input: 784 → Hidden 1: 64
  W1 = cp.random.uniform(
    low_bound,
    high_bound,
    size=(64, 784)
  )
  B1 = cp.zeros((64, 1))

  # Hidden 1: 64 → Hidden 2: 64
  W2 = cp.random.uniform(
    low_bound,
    high_bound,
    size=(64, 64)
  )
  B2 = cp.zeros((64, 1))

  # Hidden 2: 64 → Output: 10
  W3 = cp.random.uniform(
    low_bound,
    high_bound,
    size=(10, 64)
  )
  B3 = cp.zeros((10, 1))

  return W1, B1, W2, B2, W3, B3


def onehot(label: int) -> cp.ndarray:
  label = int(label)

  array = cp.zeros((10, 1))
  array[label, 0] = 1.0

  return array


def softmax(Z: cp.ndarray):
  exp_Z = cp.exp(Z - cp.max(Z))
  return exp_Z / cp.sum(exp_Z)


def relu(Z: cp.ndarray):
  return cp.maximum(0, Z)


def relu_deriv(Z: cp.ndarray):
  return (Z > 0).astype(float)


def accuracy(result: cp.ndarray, onehot_label: cp.ndarray):
  prediction = cp.argmax(result)
  actual = cp.argmax(onehot_label)

  return bool(prediction == actual)


def forward_propagation(
  X: cp.ndarray,
  W1: cp.ndarray,
  B1: cp.ndarray,
  W2: cp.ndarray,
  B2: cp.ndarray,
  W3: cp.ndarray,
  B3: cp.ndarray
):
  # Hidden layer 1: 784 → 64
  Z1 = W1 @ X + B1
  A1 = relu(Z1)

  # Hidden layer 2: 64 → 64
  Z2 = W2 @ A1 + B2
  A2 = relu(Z2)

  # Output layer: 64 → 10
  Z3 = W3 @ A2 + B3
  A3 = softmax(Z3)

  return A3, Z3, A2, Z2, A1, Z1


def backward_propagation(
  Y_onehot: cp.ndarray,
  X: cp.ndarray,
  A3: cp.ndarray,
  A2: cp.ndarray,
  Z2: cp.ndarray,
  A1: cp.ndarray,
  Z1: cp.ndarray,
  W2: cp.ndarray,
  W3: cp.ndarray
):
  # Output layer
  # Softmax + cross-entropy derivative
  dZ3 = A3 - Y_onehot
  dW3 = dZ3 @ A2.T
  dB3 = dZ3

  # Hidden layer 2
  dA2 = W3.T @ dZ3
  dZ2 = dA2 * relu_deriv(Z2)
  dW2 = dZ2 @ A1.T
  dB2 = dZ2

  # Hidden layer 1
  dA1 = W2.T @ dZ2
  dZ1 = dA1 * relu_deriv(Z1)
  dW1 = dZ1 @ X.T
  dB1 = dZ1

  return dW1, dB1, dW2, dB2, dW3, dB3


def update_params(
  alpha,
  W1, B1,
  W2, B2,
  W3, B3,
  dW1, dB1,
  dW2, dB2,
  dW3, dB3
):
  W1 = W1 - alpha * dW1
  B1 = B1 - alpha * dB1

  W2 = W2 - alpha * dW2
  B2 = B2 - alpha * dB2

  W3 = W3 - alpha * dW3
  B3 = B3 - alpha * dB3

  return W1, B1, W2, B2, W3, B3


def gradient_descent(
  iterations,
  alpha,
  W1, B1,
  W2, B2,
  W3, B3
):
  print("Progress:")

  for epoch in tqdm(range(iterations)):

    for i in range(tr_rows):

      # GPU data
      X = tr_norm[i].reshape(784, 1)

      # GPU one-hot vector
      Y_onehot = onehot(tr_label[i])

      A3, Z3, A2, Z2, A1, Z1 = forward_propagation(
        X,
        W1, B1,
        W2, B2,
        W3, B3
      )

      dW1, dB1, dW2, dB2, dW3, dB3 = backward_propagation(
        Y_onehot,
        X,
        A3,
        A2,
        Z2,
        A1,
        Z1,
        W2,
        W3
      )

      W1, B1, W2, B2, W3, B3 = update_params(
        alpha,
        W1, B1,
        W2, B2,
        W3, B3,
        dW1, dB1,
        dW2, dB2,
        dW3, dB3
      )

  return W1, B1, W2, B2, W3, B3


def main():
  W1, B1, W2, B2, W3, B3 = initparams()

  cont = 1

  print("Training Label:", tr_label[0])

  while cont == 1:

    print("Options ====================")
    print("- R for Train")
    print("- T for Test")
    print("- L for Load")
    print("- S for Save")

    option = input("Select an option: ").upper()

    if option == "R":

      iterations = int(input("Iterations: "))
      alpha = 0.01

      W1, B1, W2, B2, W3, B3 = gradient_descent(
        iterations,
        alpha,
        W1, B1,
        W2, B2,
        W3, B3
      )

    elif option == "T":

      index = int(input("Select Index: "))

      X = ts_norm[index].reshape(784, 1)

      A3, Z3, A2, Z2, A1, Z1 = forward_propagation(
        X,
        W1, B1,
        W2, B2,
        W3, B3
      )

      Y_onehot = onehot(ts_label[index])

      print(
        "Prediction Correct:",
        accuracy(A3, Y_onehot)
      )

      print("Probabilities:")

      # Convert individual GPU values to Python floats
      for i in range(10):
        print(f"{i}: {float(A3[i, 0])}")

      digit = int(cp.argmax(A3).get())

      print("Predicted digit:", digit)
      print("Actual label:", ts_label[index])

      # Move image back to CPU for Matplotlib
      image_grid = cp.asnumpy(
        ts_norm[index].reshape(28, 28)
      )

      plt.imshow(image_grid, cmap="gray")
      plt.axis("off")
      plt.show()

    elif option == "L":

      model_path = os.path.join(
        SCRIPT_DIR,
        "model_55K.npz"
      )

      model = cp.load(model_path)

      W1 = model["W1"]
      B1 = model["B1"]
      W2 = model["W2"]
      B2 = model["B2"]
      W3 = model["W3"]
      B3 = model["B3"]

      print("Model loaded.")

    elif option == "S":

      model_path = os.path.join(
        SCRIPT_DIR,
        "model_55K.npz"
      )

      cp.savez(
        model_path,
        W1=W1,
        B1=B1,
        W2=W2,
        B2=B2,
        W3=W3,
        B3=B3
      )

      print("Model saved.")

    else:
      print("Invalid input")

    print("Continue ====================")
    print("- 1 for Continue")
    print("- 0 to Exit")

    cont = int(input("Select an option: "))


main()