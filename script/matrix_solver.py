import numpy as np
from fractions import Fraction

def get_variable_names():
    print("\nChoose variable naming convention:")
    print("1. x, y, z")
    print("2. x₁, x₂, x₃")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1 or 2): "))
            if choice == 1:
                return ["x", "y", "z"]
            elif choice == 2:
                return ["x₁", "x₂", "x₃"]
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_coefficients(var_names):
    print(f"\nEnter coefficients for 3 equations (a{var_names[0]} + b{var_names[1]} + c{var_names[2]} = d):")
    equations = []
    
    for i in range(3):
        print(f"\nEquation {i+1}:")
        try:
            a = Fraction(input(f"Enter coefficient for {var_names[0]}: "))
            b = Fraction(input(f"Enter coefficient for {var_names[1]}: "))
            c = Fraction(input(f"Enter coefficient for {var_names[2]}: "))
            d = Fraction(input("Enter constant term: "))
            equations.append([a, b, c, d])
        except ValueError:
            print("Invalid input. Please enter numbers or fractions (e.g., 2 or 1/2)")
            return None
    
    return equations

def print_matrix(matrix):
    # Print top divider
    print("+" + "-" * 35 + "+")
    
    for row in matrix:
        print("|", end=" ")
        # Print coefficients
        for i in range(len(row) - 1):
            element = row[i]
            if isinstance(element, Fraction):
                if element.denominator == 1:
                    print(f"{element.numerator:3}", end=" ")
                else:
                    num = element.numerator
                    den = element.denominator
                    print(f"{num}/{den:3}", end=" ")
            else:
                print(f"{element:3}", end=" ")
        
        # Print vertical line and constant
        print("|", end=" ")
        element = row[-1]
        if isinstance(element, Fraction):
            if element.denominator == 1:
                print(f"{element.numerator:3}", end=" ")
            else:
                num = element.numerator
                den = element.denominator
                print(f"{num}/{den:3}", end=" ")
        else:
            print(f"{element:3}", end=" ")
        print("|")
    
    # Print bottom divider
    print("+" + "-" * 35 + "+")
    print()  # Add extra line for spacing

def is_singular(matrix):
    # Check if the coefficient matrix is singular
    A = np.array([[float(x) for x in row[:3]] for row in matrix])
    return np.linalg.det(A) == 0

def gaussian_elimination(matrix, var_names):
    n = len(matrix)
    step = 1
    
    print("\n" + "=" * 50)
    print("INITIAL AUGMENTED MATRIX".center(50))
    print("=" * 50)
    print_matrix(matrix)
    
    # Check for singular matrix
    if is_singular(matrix):
        print("\n" + "!" * 50)
        print("WARNING: The coefficient matrix is singular.".center(50))
        print("The system may have infinitely many solutions or no solution.".center(50))
        print("!" * 50 + "\n")
    
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        max_row = i
        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                max_row = j
        
        if max_row != i:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            print(f"\nStep {step}: R{i+1} ↔ R{max_row+1}")
            step += 1
            print_matrix(matrix)
        
        # Make the diagonal element 1
        if matrix[i][i] == 0:
            # Check for inconsistent system
            if matrix[i][n] != 0:
                print("\n" + "!" * 50)
                print("The system is inconsistent and has no solution.".center(50))
                print("!" * 50 + "\n")
                return None
            continue
        
        pivot = matrix[i][i]
        for j in range(i, n + 1):
            matrix[i][j] = matrix[i][j] / pivot
        print(f"\nStep {step}: R{i+1} → R{i+1}/{pivot}")
        step += 1
        print_matrix(matrix)
        
        # Eliminate other elements in the column
        for k in range(i + 1, n):
            factor = matrix[k][i]
            if factor != 0:
                for j in range(i, n + 1):
                    matrix[k][j] = matrix[k][j] - factor * matrix[i][j]
                print(f"\nStep {step}: R{k+1} → R{k+1} - {factor}×R{i+1}")
                step += 1
                print_matrix(matrix)
    
    # Check for inconsistent system or infinitely many solutions
    for i in range(n):
        if all(matrix[i][j] == 0 for j in range(n)) and matrix[i][n] != 0:
            print("\n" + "!" * 50)
            print("The system is inconsistent and has no solution.".center(50))
            print("!" * 50 + "\n")
            return None
    
    print("\n" + "=" * 50)
    print("MATRIX IN ROW ECHELON FORM".center(50))
    print("=" * 50)
    print_matrix(matrix)
    
    print("\n" + "=" * 50)
    print("BACK SUBSTITUTION STEPS".center(50))
    print("=" * 50)
    
    # Back substitution
    x = [Fraction(0, 1) for _ in range(n)]
    for i in range(n - 1, -1, -1):
        if matrix[i][i] == 0:
            if matrix[i][n] == 0:
                print("\n" + "!" * 50)
                print("The system has infinitely many solutions.".center(50))
                print("!" * 50 + "\n")
                return None
            else:
                print("\n" + "!" * 50)
                print("The system is inconsistent and has no solution.".center(50))
                print("!" * 50 + "\n")
                return None
        
        x[i] = matrix[i][n]
        substitution = f"{var_names[i]} = {x[i]}"
        
        for j in range(i + 1, n):
            if matrix[i][j] != 0:
                x[i] = x[i] - matrix[i][j] * x[j]
                if matrix[i][j] * x[j] > 0:
                    substitution += f" - {matrix[i][j]}×{x[j]}"
                else:
                    substitution += f" + {-matrix[i][j]}×{x[j]}"
        
        x[i] = x[i] / matrix[i][i]
        print(f"\nStep {step}: {substitution}")
        print(f"     {'=' * 20}")
        print(f"     {var_names[i]} = {x[i]}")
        step += 1
    
    return x

def main():
    print("\n" + "=" * 50)
    print("SYSTEM OF EQUATIONS SOLVER".center(50))
    print("=" * 50)
    
    var_names = get_variable_names()
    equations = get_coefficients(var_names)
    if equations is None:
        return
    
    matrix = [row[:] for row in equations]  # Create a copy of the equations
    
    solution = gaussian_elimination(matrix, var_names)
    if solution is not None:
        print("\n" + "=" * 50)
        print("FINAL SOLUTION".center(50))
        print("=" * 50)
        for i in range(len(solution)):
            print(f"\n     {var_names[i]} = {solution[i]}")
        print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 