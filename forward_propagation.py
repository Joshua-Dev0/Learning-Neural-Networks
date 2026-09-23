import numpy as np

def softmax(Z: np.ndarray):
  exp_Z = np.exp(Z - np.max(Z))
  return exp_Z / np.sum(exp_Z)

def relu(Z: np.ndarray):
  return np.maximum(0, Z)


def forward_propagation(X: np.ndarray, W1: np.ndarray, B1: np.ndarray, W2: np.ndarray, B2: np.ndarray):
  # Pass 1: Hidden Layer calculation (Outputs a 64x1 matrix)
  Z1 = (W1 @ X) + B1  # ([64x784] * [784x1]) + [64x1] = [64x1]
  A1 = relu(Z1)  
  
  # Pass 2: Output Layer calculation (Outputs a 10x1 matrix)
  Z2 = (W2 @ A1) + B2  # ([10x64] * [64, 1]) + [10x1] = [10x1]
  A2 = softmax(Z2)     # Scales the 10 raw digits scores into probabilities
  
  return A2, Z2, A1, Z1

  
  dZ2 = A2 - Y_onehot    #Cost + softmax function derivative
  dW2 = dZ2 @ A1.T    # for each image trained, dW2 = (1/m) * (dZ2 @ A1.T) for batch size m
  dB2 = dZ2
  
  dA1 = W2.T @ dZ2
  dZ1 = dA1 * relu_deriv(Z1)
  dW1 = dZ1 @ X.T
  dB1 = dZ1
  
  return dW2, dB2, dW1, dB1

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