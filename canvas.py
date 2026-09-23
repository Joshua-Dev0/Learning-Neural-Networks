import os
import tkinter as tk
import numpy as np

from forward_propagation import forward_propagation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "model.npz")

model = np.load(model_path)
W1 = model["W1"]
B1 = model["B1"]
W2 = model["W2"]
B2 = model["B2"]

class NativeMNISTCanvas:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 Native MNIST Drawer (0-255)")
        
        self.grid_size = 28
        self.pixel_scale = 16
        self.canvas_dim = self.grid_size * self.pixel_scale
        
        self.pixel_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        self.canvas = tk.Canvas(root, width=self.canvas_dim, height=self.canvas_dim, bg="black")
        self.canvas.pack(pady=10)
        
        self.draw_grid_lines()
            
        self.canvas.bind("<B1-Motion>", self.paint_brush)
        self.canvas.bind("<Button-1>", self.paint_brush)
        
        self.btn_process = tk.Button(root, text="Inference", command=self.output_vector)
        self.btn_process.pack(side=tk.LEFT, padx=30, pady=10)
        
        self.btn_clear = tk.Button(root, text="Clear Canvas", command=self.clear_canvas)
        self.btn_clear.pack(side=tk.RIGHT, padx=30, pady=10)

    def draw_grid_lines(self):
        for i in range(self.grid_size):
            self.canvas.create_line(0, i * self.pixel_scale, self.canvas_dim, i * self.pixel_scale, fill="#121212")
            self.canvas.create_line(i * self.pixel_scale, 0, i * self.pixel_scale, self.canvas_dim, fill="#121212")

    def paint_brush(self, event):
        col = event.x // self.pixel_scale
        row = event.y // self.pixel_scale
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r_target, c_target = row + dr, col + dc
                
                if 0 <= r_target < self.grid_size and 0 <= c_target < self.grid_size:
                    if dr == 0 and dc == 0:
                        intensity = 255.0
                    elif abs(dr) == 1 and abs(dc) == 1:
                        intensity = max(self.pixel_grid[r_target, c_target], 100.0)
                    else:
                        intensity = max(self.pixel_grid[r_target, c_target], 180.0)
                    
                    self.pixel_grid[r_target, c_target] = intensity
                    
                    hex_color = f"#{int(intensity):02x}{int(intensity):02x}{int(intensity):02x}"
                    
                    x1, y1 = c_target * self.pixel_scale, r_target * self.pixel_scale
                    x2, y2 = x1 + self.pixel_scale, y1 + self.pixel_scale
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=hex_color, outline="#121212")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.pixel_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.draw_grid_lines()

    def output_vector(self):
        flat_array = self.pixel_grid.flatten()
        X = flat_array.reshape(784, 1)
        
        A2, Z2, A1, Z1 = forward_propagation(X, W1, B1, W2, B2)
        digit = np.argmax(A2)
        print("Predicted digit:", digit)
        

        global native_mnist_input
        native_mnist_input = X

native_mnist_input = None
root = tk.Tk()
app = NativeMNISTCanvas(root)
root.mainloop()
