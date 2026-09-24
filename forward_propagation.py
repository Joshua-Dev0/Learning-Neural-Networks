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