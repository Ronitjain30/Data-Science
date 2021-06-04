import numpy as np

print("Creating a square matrix :-")
rows_and_columns = int(input("Enter the number of rows and columns : "))

print("Enter the matrix values in a single line (separated by space) : ")

matrix_values = list(map(int, input().split()))
if (len(matrix_values) != rows_and_columns**2):
    print("Total number of matrix values should be equal to", rows_and_columns**2)
    exit()

matrix = np.array(matrix_values).reshape(rows_and_columns, rows_and_columns)

print("Input Matrix :-")
print(matrix)

array_of_left_diagonal = matrix.diagonal().tolist()

np.fill_diagonal(matrix, np.fliplr(matrix).diagonal())
np.fill_diagonal(np.fliplr(matrix), array_of_left_diagonal)

print("Output Matrix :-")
print(matrix)
