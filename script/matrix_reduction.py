#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple
from fractions import Fraction

def is_row_echelon(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is in row echelon form.
    
    Args:
        matrix: Input matrix as a numpy array
        
    Returns:
        True if matrix is in REF, False otherwise
    """
    m, n = matrix.shape
    last_pivot_col = -1
    
    for row in range(m):
        # Find the first non-zero element in the row
        pivot_col = -1
        for col in range(n):
            if abs(matrix[row, col]) > 1e-10:
                pivot_col = col
                break
        
        # If this row is all zeros, all rows below must also be all zeros
        if pivot_col == -1:
            for r in range(row + 1, m):
                if any(abs(matrix[r, col]) > 1e-10 for col in range(n)):
                    return False
            return True
        
        # Pivot must be to the right of previous pivot
        if pivot_col <= last_pivot_col:
            return False
        
        last_pivot_col = pivot_col
        
        # All elements below pivot must be zero
        for r in range(row + 1, m):
            if abs(matrix[r, pivot_col]) > 1e-10:
                return False
    
    return True

def is_reduced_row_echelon(matrix: np.ndarray) -> bool:
    """
    Check if a matrix is in reduced row echelon form.
    
    Args:
        matrix: Input matrix as a numpy array
        
    Returns:
        True if matrix is in RREF, False otherwise
    """
    if not is_row_echelon(matrix):
        return False
    
    m, n = matrix.shape
    
    for row in range(m):
        # Find the pivot
        pivot_col = -1
        for col in range(n):
            if abs(matrix[row, col]) > 1e-10:
                pivot_col = col
                break
        
        if pivot_col == -1:
            continue
        
        # Pivot must be 1
        if abs(matrix[row, pivot_col] - 1) > 1e-10:
            return False
        
        # All elements above pivot must be zero
        for r in range(row):
            if abs(matrix[r, pivot_col]) > 1e-10:
                return False
    
    return True

def to_row_echelon(matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a matrix to row echelon form (REF) using Gaussian elimination.
    
    Args:
        matrix: Input matrix as a numpy array
        
    Returns:
        Tuple of (Matrix in row echelon form, List of steps taken)
    """
    if is_row_echelon(matrix):
        return matrix, ["Matrix is already in Row Echelon Form"]
    
    m, n = matrix.shape
    ref_matrix = matrix.copy().astype(float)
    steps = []
    
    # Forward elimination
    for col in range(min(m, n)):
        # Find the row with the largest absolute value in current column
        max_row = np.argmax(np.abs(ref_matrix[col:, col])) + col
        
        # Swap rows if necessary
        if max_row != col:
            ref_matrix[[col, max_row]] = ref_matrix[[max_row, col]]
            steps.append(f"Row Swap: R{col+1} ↔ R{max_row+1}")
            steps.append(matrix_to_fraction_string(ref_matrix))
        
        # Skip if the pivot is zero
        if ref_matrix[col, col] == 0:
            steps.append(f"Pivot in column {col+1} is zero, moving to next column")
            continue
            
        # Eliminate all elements below the pivot
        for row in range(col + 1, m):
            if ref_matrix[row, col] != 0:
                factor = ref_matrix[row, col] / ref_matrix[col, col]
                ref_matrix[row, col:] -= factor * ref_matrix[col, col:]
                steps.append(f"Row Operation: R{row+1} = R{row+1} - ({Fraction(factor).limit_denominator()}) * R{col+1}")
                steps.append(matrix_to_fraction_string(ref_matrix))
    
    return ref_matrix, steps

def to_reduced_row_echelon(matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a matrix to reduced row echelon form (RREF) using Gauss-Jordan elimination.
    
    Args:
        matrix: Input matrix as a numpy array
        
    Returns:
        Tuple of (Matrix in reduced row echelon form, List of steps taken)
    """
    if is_reduced_row_echelon(matrix):
        return matrix, ["Matrix is already in Reduced Row Echelon Form"]
    
    m, n = matrix.shape
    rref_matrix = matrix.copy().astype(float)
    steps = []
    
    # Forward elimination (same as REF)
    for col in range(min(m, n)):
        max_row = np.argmax(np.abs(rref_matrix[col:, col])) + col
        
        if max_row != col:
            rref_matrix[[col, max_row]] = rref_matrix[[max_row, col]]
            steps.append(f"Row Swap: R{col+1} ↔ R{max_row+1}")
            steps.append(matrix_to_fraction_string(rref_matrix))
        
        if rref_matrix[col, col] == 0:
            steps.append(f"Pivot in column {col+1} is zero, moving to next column")
            continue
            
        # Make the pivot 1
        if rref_matrix[col, col] != 1:
            factor = rref_matrix[col, col]
            rref_matrix[col, :] /= factor
            steps.append(f"Row Operation: R{col+1} = R{col+1} / {Fraction(factor).limit_denominator()}")
            steps.append(matrix_to_fraction_string(rref_matrix))
        
        # Eliminate all elements above and below the pivot
        for row in range(m):
            if row != col and rref_matrix[row, col] != 0:
                factor = rref_matrix[row, col]
                rref_matrix[row, :] -= factor * rref_matrix[col, :]
                steps.append(f"Row Operation: R{row+1} = R{row+1} - ({Fraction(factor).limit_denominator()}) * R{col+1}")
                steps.append(matrix_to_fraction_string(rref_matrix))
    
    return rref_matrix, steps

def matrix_to_fraction_string(matrix: np.ndarray) -> str:
    """
    Convert a numpy matrix to a string with fractions.
    
    Args:
        matrix: Input matrix as a numpy array
        
    Returns:
        String representation of the matrix with fractions
    """
    rows = []
    for row in matrix:
        row_str = []
        for elem in row:
            if abs(elem) < 1e-10:  # Handle very small numbers as zero
                row_str.append("0")
            else:
                frac = Fraction(elem).limit_denominator()
                if frac.denominator == 1:
                    row_str.append(str(frac.numerator))
                else:
                    row_str.append(f"{frac.numerator}/{frac.denominator}")
        rows.append("[" + " ".join(row_str) + "]")
    return "\n".join(rows)

def print_matrix(matrix: np.ndarray, title: str = ""):
    """
    Print a matrix with a title in a formatted way.
    
    Args:
        matrix: Matrix to print
        title: Optional title for the matrix
    """
    if title:
        print(f"\n{title}:")
    print(matrix_to_fraction_string(matrix))
    print()

def print_steps(steps: List[str], title: str):
    """
    Print the steps taken during matrix transformation.
    
    Args:
        steps: List of steps taken
        title: Title for the steps section
    """
    print(f"\n{title}:")
    print("=" * 50)
    step_count = 0
    for i, step in enumerate(steps):
        if i % 2 == 0:  # Operation description
            step_count += 1
            print(f"\nStep {step_count}: {step}")
            print("-" * 30)
        else:  # Matrix state
            print("\nMatrix after operation:")
            print(step)
            print("-" * 30)
    print("=" * 50)

def get_custom_matrix() -> np.ndarray:
    """
    Get a custom matrix from user input.
    
    Returns:
        Custom matrix as a numpy array
    """
    while True:
        try:
            rows_input = input("Enter number of rows (or 'exit' to quit): ")
            if rows_input.lower() == 'exit':
                exit(0)
            rows = int(rows_input)
            
            cols_input = input("Enter number of columns (or 'exit' to quit): ")
            if cols_input.lower() == 'exit':
                exit(0)
            cols = int(cols_input)
            
            if rows > 0 and cols > 0:
                break
            print("Please enter positive integers for dimensions")
        except ValueError:
            print("Please enter valid integers")
    
    print("\nEnter matrix elements row by row:")
    print("Use spaces to separate elements")
    print("Press Enter after each row")
    print("Press Enter twice when done")
    print("Type 'exit' at any time to quit\n")
    
    matrix_data = []
    while True:
        row_input = input()
        if row_input.lower() == 'exit':
            exit(0)
        if not row_input:  # Empty line means end of input
            break
        try:
            row = [float(x) for x in row_input.split()]
            if len(row) != cols:
                print(f"Expected {cols} elements, got {len(row)}")
                continue
            matrix_data.append(row)
        except ValueError:
            print("Please enter valid numbers separated by spaces")
    
    if len(matrix_data) != rows:
        print(f"Expected {rows} rows, got {len(matrix_data)}")
        return get_custom_matrix()
    
    return np.array(matrix_data)

def main():
    # Example matrices
    matrices = [
        # 1. Simple 2x2 matrix
        np.array([[-3, -9, 9, -5],
                  [-9, -9, -10, 8],
                  [1, 2, 3, 4]]),
        
        # 2. 3x3 matrix with zero pivot
        np.array([[-3, -9, 9, -5],
                  [-9, -9, -10, 8],
                  [1, 2, 3, 4]]),
        
        # 3. 3x4 matrix representing a system of equations
        np.array([[2, 1, -1, 8],
                  [-3, -1, 2, -11],
                  [-2, 1, 2, -3]]),
        
        # 4. 3x3 matrix already in REF
        np.array([[-3, -9, 9, -5],
                  [-9, -9, -10, 8],
                  [1, 2, 3, 4]]),
        
        # 5. 3x3 matrix already in RREF
        np.array([[1, 0, 0, 2],
                  [0, 1, 0, 3],
                  [0, 0, 1, 4]]),
        
        # 6. 4x4 matrix with fractions
        np.array([[1, 1/2, 1/3, 1/4, 1],
                  [1/2, 1/3, 1/4, 1/5, 1],
                  [1/3, 1/4, 1/5, 1/6, 1],
                  [1/4, 1/5, 1/6, 1/7, 1]]),
        
        # 7. 3x3 matrix with negative numbers
        np.array([[1, 1/2, 1/3, 1/4, 1],
                  [1/2, 1/3, 1/4, 1/5, 1],
                  [1/3, 1/4, 1/5, 1/6, 1],
                  [1/4, 1/5, 1/6, 1/7, 1]]),
        
        # 8. 2x4 matrix
        np.array([[1, 1/2, 1/3, 1/4, 1],
                  [1/2, 1/3, 1/4, 1/5, 1],
                  [1/3, 1/4, 1/5, 1/6, 1],
                  [1/4, 1/5, 1/6, 1/7, 1]]),
        
        # 9. 3x3 matrix with all ones
        np.array([[1, 1/2, 1/3, 1/4, 1],
                  [1/2, 1/3, 1/4, 1/5, 1],
                  [1/3, 1/4, 1/5, 1/6, 1],
                  [1/4, 1/5, 1/6, 1/7, 1]]),
        
        # 10. 3x3 identity matrix
        np.array([[1, 1/2, 1/3],
                  [1/2, 1/3, 1/4],
                  [1/3, 1/4, 1/5],
                  [1/4, 1/5, 1/6]])
    ]
    
    # Get user choice
    while True:
        try:
            choice_input = input("Press 1 for predefined matrices, 2 for custom matrix, or 'exit' to quit: ")
            if choice_input.lower() == 'exit':
                exit(0)
            choice = int(choice_input)
            if choice in [1, 2]:
                break
            print("Please enter 1 or 2")
        except ValueError:
            print("Please enter a valid number")
    
    if choice == 1:
        # Print available matrices
        print("\nAvailable matrices:")
        for i, matrix in enumerate(matrices, 1):
            print(f"\nMatrix {i}:")
            print_matrix(matrix)
        
        # Get matrix selection
        while True:
            try:
                matrix_choice_input = input("\nEnter the matrix number you want to solve (1-10) or 'exit' to quit: ")
                if matrix_choice_input.lower() == 'exit':
                    exit(0)
                matrix_choice = int(matrix_choice_input)
                if 1 <= matrix_choice <= 10:
                    break
                print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        matrix = matrices[matrix_choice - 1]
    else:
        matrix = get_custom_matrix()
    
    print(f"\n{'='*20} Matrix {'='*20}")
    print_matrix(matrix, "Original Matrix")
    
    # Check if matrix is already in RREF
    if is_reduced_row_echelon(matrix):
        print("Matrix is already in Reduced Row Echelon Form!")
        return
    
    # Convert to Row Echelon Form
    ref_matrix, ref_steps = to_row_echelon(matrix)
    print_steps(ref_steps, "Steps to Row Echelon Form")
    print_matrix(ref_matrix, "Final Row Echelon Form")
    
    # Only convert to RREF if not already in RREF
    if not is_reduced_row_echelon(ref_matrix):
        rref_matrix, rref_steps = to_reduced_row_echelon(ref_matrix)
        print_steps(rref_steps, "Steps to Reduced Row Echelon Form")
        print_matrix(rref_matrix, "Final Reduced Row Echelon Form")
    else:
        print("Matrix is already in Reduced Row Echelon Form!")
    
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main() 