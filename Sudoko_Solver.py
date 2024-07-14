def solve_sudoku(board):
    empty_cell = find_empty(board)
    if not empty_cell:
        return True
    
    row, col = empty_cell
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    
    return False

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board, row, col, num):
    # Check row
    for j in range(len(board[0])):
        if board[row][j] == num:
            return False
    
    # Check column
    for i in range(len(board)):
        if board[i][col] == num:
            return False
    
    # Check 3x3 sub-grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(start_row, start_row + 3):
        for j in range(start_col, start_col + 3):
            if board[i][j] == num:
                return False
    
    return True

def print_board(board):
    for row in board:
        print(row)
def take_input():
    print("Enter the Sudoku board (use 0 to represent empty cells):")
    board = []
    for i in range(9):
        row = list(map(int, input().split()))
        board.append(row)
    return board
"""board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]"""
board = take_input()
print("Sudoku board to solve:")
print_board(board)

if solve_sudoku(board):
    print("\nSolution:")
    print_board(board)
else:
    print("No solution exists")
