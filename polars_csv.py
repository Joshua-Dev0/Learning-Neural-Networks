import os
import polars as pl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "mnist_test.csv")

df = pl.read_csv(csv_path)

# ---- Display the first column ----
# first_column = df.select(pl.nth(0))
# print(first_column)

# display first row
# first_row_values = df.row(0)
# print(first_row_values)

# 2. Slice it to skip index 0 (the first column's number)
# pixels_only = first_row_values[1:]

# 3. Print it out (This will show exactly 784 pixel values)
# print(pixels_only)
# print("Number of items:", len(pixels_only))  # Should print 784

#-------------------------------------------------------------------------------------------
# First column = labels
labels = df[:, 0]
print("Labels:")
print(labels)

# # Everything except first column = pixels
# pixels = df[:, 1:]
# print("\nPixels:")
# print(pixels)

#First row of pixels
row = 3
first_row_pixels = df[row, 1:]
print("First Row Pixels:")
print(first_row_pixels)

# num_rows = df.height
# print("Number of columns:", num_rows)  # For MNIST, this will be 10000
# print("Pixels shape:", pixels.shape)
