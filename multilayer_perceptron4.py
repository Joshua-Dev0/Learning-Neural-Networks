import cupy as cp

# Config
input_size = 784
hiddenl_size = 64
hiddenl_count = 2
output_size = 10

class Layer:
  def __init__(self, input_size, output_size):
    self.weights = cp.random.uniform(-0.1, 0.1, size=(output_size, input_size))
    self.bias = cp.zeros((output_size, 1))

def initparams():
  layers = []

  for i in range(hiddenl_count + 1):
    if i == 0:
      layers.append(Layer(input_size, hiddenl_size))

    elif i < hiddenl_count:
      layers.append(Layer(hiddenl_size, hiddenl_size))

    else:
      layers.append(Layer(hiddenl_size, output_size))

  return layers

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


# Softmax not implemented
def forward_propagation(X, layers):
  A = [X]
  Z = []
  
  for i in range(hiddenl_count + 1):
    Z = layers[i].weights @ A[i] + layers[i].bias
    A = relu(Z)
    
    Zs.append(Z)
    As.append(A)

  return As, Zs

def backward_propagation(Y_onehot: cp.ndarray, A, Z, layers):
  dW = []
  dB = []
  
  for i in reversed(range(hiddenl_count + 1)):
    if (i < hiddenl_count):
      dA = layers[i].weights.T @ dZ
      
    if (i == hiddenl_count):
      dZ = A[i] - Y_onehot
    elif (i < hiddenl_count):
      dZ = dA * relu_deriv(Z[i])
      
    dW = dZ @ A[i].T
    dB = dZ

  return dW, dB

def update_params(alpha, layers, dW, dB):
  
  for i in range(layers):
    layers[i].weights = layers[i].weights - alpha * dW[i]
    layers[i].bias = layers[i].bias - alpha * dB[i]

  return layers

# def gradient_descent(epoch, alpha, layers):

#   for e in range(epoch):
#     for i in range(tr_rows):

#       # GPU data
#       X = tr_norm[i].reshape(784, 1)

#       # GPU one-hot vector
#       Y_onehot = onehot(tr_label[i])

#       A3, Z3, A2, Z2, A1, Z1 = forward_propagation(
#         X,
#         W1, B1,
#         W2, B2,
#         W3, B3
#       )

#       dW1, dB1, dW2, dB2, dW3, dB3 = backward_propagation(
#         Y_onehot,
#         X,
#         A3,
#         A2,
#         Z2,
#         A1,
#         Z1,
#         W2,
#         W3
#       )

#       W1, B1, W2, B2, W3, B3 = update_params(
#         alpha,
#         W1, B1,
#         W2, B2,
#         W3, B3,
#         dW1, dB1,
#         dW2, dB2,
#         dW3, dB3
#       )

#   return W1, B1, W2, B2, W3, B3

# def main():
#   layers = initparams()

#   cont = 1

#   print("Training Label:", tr_label[0])

#   while cont == 1:

#     print("Options ====================")
#     print("- R for Train")
#     print("- T for Test")
#     print("- L for Load")
#     print("- S for Save")

#     option = input("Select an option: ").upper()

#     if option == "R":

#       iterations = int(input("Iterations: "))
#       alpha = 0.01

#       W1, B1, W2, B2, W3, B3 = gradient_descent(
#         iterations,
#         alpha,
#         W1, B1,
#         W2, B2,
#         W3, B3
#       )

#     elif option == "T":

#       index = int(input("Select Index: "))

#       X = ts_norm[index].reshape(784, 1)

#       A3, Z3, A2, Z2, A1, Z1 = forward_propagation(
#         X,
#         W1, B1,
#         W2, B2,
#         W3, B3
#       )

#       Y_onehot = onehot(ts_label[index])

#       print(
#         "Prediction Correct:",
#         accuracy(A3, Y_onehot)
#       )

#       print("Probabilities:")

#       # Convert individual GPU values to Python floats
#       for i in range(10):
#         print(f"{i}: {float(A3[i, 0])}")

#       digit = int(cp.argmax(A3).get())

#       print("Predicted digit:", digit)
#       print("Actual label:", ts_label[index])

#       # Move image back to CPU for Matplotlib
#       image_grid = cp.asnumpy(
#         ts_norm[index].reshape(28, 28)
#       )

#       plt.imshow(image_grid, cmap="gray")
#       plt.axis("off")
#       plt.show()

#     elif option == "L":

#       model_path = os.path.join(
#         SCRIPT_DIR,
#         "model_55K.npz"
#       )

#       model = cp.load(model_path)

#       W1 = model["W1"]
#       B1 = model["B1"]
#       W2 = model["W2"]
#       B2 = model["B2"]
#       W3 = model["W3"]
#       B3 = model["B3"]

#       print("Model loaded.")

#     elif option == "S":

#       model_path = os.path.join(
#         SCRIPT_DIR,
#         "model_55K.npz"
#       )

#       cp.savez(
#         model_path,
#         W1=W1,
#         B1=B1,
#         W2=W2,
#         B2=B2,
#         W3=W3,
#         B3=B3
#       )

#       print("Model saved.")

#     else:
#       print("Invalid input")

#     print("Continue ====================")
#     print("- 1 for Continue")
#     print("- 0 to Exit")

#     cont = int(input("Select an option: "))


# main()