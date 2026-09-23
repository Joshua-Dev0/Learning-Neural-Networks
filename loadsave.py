import os
import numpy as np

W1=10
B1=20
W2=30
B2=40

np.savez(
  "model.npz",
  W1=W1,
  B1=B1,
  W2=W2,
  B2=B2
)

data = np.load("model.npz")

print("W1: ", data["W1"])
print("B1: ", data["B1"])
print("W2: ", data["W2"])
print("B2: ", data["B2"])