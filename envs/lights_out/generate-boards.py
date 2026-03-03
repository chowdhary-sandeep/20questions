import numpy as np
import random
from copy import deepcopy
from itertools import combinations
from datasets import Dataset
import os

def build_toggle_matrix(n):
    """
    Build the n^2 x n^2 toggle matrix A over GF(2) for Lights Out rules.
    Each row corresponds to pressing one of the n*n cells.
    """
    size = n * n
    A = np.zeros((size, size), dtype=np.uint8)
    
    def idx(i, j):
        return i * n + j
    
    for i in range(n):
        for j in range(n):
            r = idx(i, j)
            # pressing this toggles itself
            A[r, idx(i, j)] = 1
            # neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    A[r, idx(ni, nj)] = 1
    return A % 2

def gf2_rank(mat):
    """Compute rank of a matrix over GF2 via Gaussian elimination."""
    M = mat.copy().astype(np.uint8)
    rows, cols = M.shape
    rank = 0
    row = 0
    for col in range(cols):
        # find pivot
        pivot = None
        for r in range(row, rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        # swap if necessary
        if pivot != row:
            M[[row, pivot], :] = M[[pivot, row], :]
        # eliminate below
        for r in range(row+1, rows):
            if M[r, col] == 1:
                M[r, :] ^= M[row, :]
        row += 1
        rank += 1
        if row == rows:
            break
    return rank

def gf2_solve(A, b):
    """
    Solve A x = b over GF2.
    Returns a particular solution x0 (size vector), and a basis for nullspace (each basis vector).
    Uses numpy row-reduction.
    """
    # Augment A | b
    A = A.copy().astype(np.uint8)
    b = b.copy().astype(np.uint8).reshape(-1, 1)
    n_rows, n_cols = A.shape
    
    # Build augmented matrix
    M = np.concatenate([A, b], axis=1)
    
    # Perform Gaussian elimination on columns 0..n_cols-1
    row = 0
    pivots = []
    for col in range(n_cols):
        # find pivot at or below row
        pivot = None
        for r in range(row, n_rows):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        # swap to current row
        if pivot != row:
            M[[row, pivot], :] = M[[pivot, row], :]
        pivots.append(col)
        # eliminate in other rows
        for r in range(n_rows):
            if r != row and M[r, col] == 1:
                M[r, :] ^= M[row, :]
        row += 1
        if row == n_rows:
            break
    
    # Check consistency: rows of zeros in A but nonzero in b column => no solution
    for r in range(row, n_rows):
        if (M[r, :n_cols] == 0).all() and M[r, n_cols] == 1:
            return None, []  # no solution

    # Extract a particular solution x0
    # Initialize x0 = zeros
    x0 = np.zeros((n_cols,), dtype=np.uint8)
    # For pivot columns, set x0[col] = value in augmented part
    for r, col in enumerate(pivots):
        # row r corresponds to pivot in column col
        x0[col] = M[r, n_cols]  # since row has been reduced

    # Nullspace basis: for each non-pivot column, create a basis vector
    free_cols = [c for c in range(n_cols) if c not in pivots]
    null_basis = []
    for free in free_cols:
        z = np.zeros((n_cols,), dtype=np.uint8)
        z[free] = 1
        # For each pivot row, see if that pivot column depends on this free var
        for r, pc in enumerate(pivots):
            if M[r, free] == 1:
                z[pc] = 1
        null_basis.append(z)
    
    return x0, null_basis

def canonical_solutions_linear(initial_state, n):
    """
    Compute minimal solutions via linear algebra only.
    Returns:
      - list of solutions, each solution is a list of (i,j) toggles
      - minimal number of steps
    """
    A = build_toggle_matrix(n=n)
    size = n * n
    b = np.array(initial_state, dtype=np.uint8).reshape(size)
    x0, null_basis = gf2_solve(A, b)
    if x0 is None:
        raise ValueError("State not solvable (should not happen for 3×3)")
    
    # If no null basis, only one solution
    sols = []
    def vec_to_coords(x_vec):
        return [(k // n, k % n) for k, bit in enumerate(x_vec) if bit == 1]

    if not null_basis:
        sols = [vec_to_coords(x0)]
        min_steps = int(x0.sum())
        return sols, min_steps

    # Else there are multiple solutions: all x = x0 + linear combinations of null_basis
    # Find minimal among those
    min_weight = size + 1
    sols = []
    # There are 2^len(null_basis) combos
    for bits in range(1 << len(null_basis)):
        x = x0.copy()
        for j in range(len(null_basis)):
            if (bits >> j) & 1:
                x ^= null_basis[j]
        w = int(x.sum())
        if w < min_weight:
            sols = [vec_to_coords(x)]
            min_weight = w
        elif w == min_weight:
            sols.append(vec_to_coords(x))
    return sols, min_steps if not null_basis else min(min_steps, min_weight)

def random_board(num_lights_on, n):
    total = n * n
    ones = random.sample(range(total), num_lights_on)
    board = [[0]*n for _ in range(n)]
    for p in ones:
        i, j = divmod(p, n)
        board[i][j] = 1
    return board

def generate_dataset_linear(num_examples, n, seed=0):
    random.seed(seed)
    seen = set()
    data = []
    while len(data) < num_examples:
        min_on, max_on = 1, n*n
        k = random.randint(min_on, max_on)
        board = random_board(k, n)
        key = tuple(tuple(row) for row in board)
        if key in seen:
            continue
        seen.add(key)
        sols, min_steps = canonical_solutions_linear(board, n)
        data.append({
            "initial_state": board,
            "minimal_solutions": sols,
            "minimal_solution_steps": min_steps
        })
    return data

if __name__ == "__main__":
    if not os.getenv("HF_USERNAME"):
        raise ValueError("Set HF_USERNAME environment variable to your Hugging Face username.")
    print("WARNING: Make sure you are logged in to HuggingFace in your current environment via `huggingface-cli login`")
    ds = generate_dataset_linear(num_examples=300, n=3, seed=42)
    for i, item in enumerate(ds[:10]):
        print(f"Example {i}:")
        print(" initial:", item["initial_state"])
        print(" min_steps:", item["minimal_solution_steps"])
        print(" sols:", item["minimal_solutions"])
        print()
    hf_ds = Dataset.from_list(ds)
    hf_ds = hf_ds.train_test_split(test_size=0.1)
    hf_ds.push_to_hub(f"{os.getenv('HF_USERNAME')}/lights-out-3x3", private=False)
