# MLP From Scratch Playground 🧠🎒

Welcome to the **MLP From Scratch Playground**! This repository is a dedicated sandbox created to learn, experiment, and implement a **Multilayer Perceptron (MLP)** entirely from scratch. 

By bypassing heavy deep learning frameworks like PyTorch or TensorFlow, this project focuses on implementing the foundational mathematics—matrix multiplications, forward propagation, activation functions, backpropagation, and weight updates—using raw numerical and data processing libraries.

---

## 🛠️ The Tech Stack

To keep the playground focused on foundational engineering, the implementation relies on three core libraries:

*   **[NumPy](https://numpy.org/)**: Handles the low-level, high-performance matrix algebra (`A @ B`), weight initializations (Xavier/He), and element-wise array operations.
*   **[Polars](https://pola.rs/)**: A lightning-fast DataFrames library written in Rust. Used for highly efficient data loading, preprocessing, tokenization, and input normalization pipelines.
*   **[Matplotlib](https://matplotlib.org/)**: Provides visual tracking of training dynamics, plotting loss curves, weight distribution shifts, and decision boundaries.

---

## 🚀 Core Learning Objectives & Features

This repository serves as an educational journal for mastering neural network mechanics:

*   **Weight Dynamics & Initialization:** Experiments with why weights shouldn't just be between 0 and 1, implementing random initialization centered around zero (**He** for ReLU, **Xavier** for Sigmoid/Tanh).
*   **The Forward-Backward Loop:** Writing the explicit `for` loop training cycles from scratch, including tracking activations, computing gradients, and implementing structural **Weight Decay (L2 Regularization)** to keep weights from exploding into `NaN`.
*   **Lossless Vectorization:** Maximizing calculation efficiency by organizing structural features into tight matrices rather than slow Python loops.
*   **Predictive Foundations:** Demystifying how deep learning models (all the way up to complex autoregressive prediction models like GPT) fundamentally rely on basic, sequential matrix operations and probability distributions.

---

## 📊 Visualizing Results

During training, `matplotlib` automatically outputs diagnostic curves to ensure the model isn't breaking:
1. **Loss Curve:** Tracks how quickly the prediction error approaches zero.
2. **Weight Histograms:** Monitors the magnitude of your weights to catch **exploding gradients** before they hit maximum floating-point boundaries.
