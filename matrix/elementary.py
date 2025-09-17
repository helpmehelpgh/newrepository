# matrix/elementary.py
# Elementary row operations with PyTorch tensors (0-based row indices).

import torch

def _as_float_tensor(A):
    """Return a float tensor copy (keeps inputs like lists/np arrays usable)."""
    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float64)
    elif A.dtype not in (torch.float64, torch.float32):
        A = A.to(torch.float64)
    return A.clone()

def rowswap(A, i: int, j: int):
    """Swap rows i and j."""
    M = _as_float_tensor(A)
    M[[i, j], :] = M[[j, i], :]
    return M

def rowscale(A, i: int, factor: float):
    """Scale row i by `factor`."""
    M = _as_float_tensor(A)
    M[i, :] = factor * M[i, :]
    return M

def rowreplacement(A, i: int, j: int, j_scale: float = 1.0, k_scale: float = 1.0):
    """
    Row replacement: R_i := j_scale * R_i + k_scale * R_j
    (Keeps R_j unchanged).
    """
    M = _as_float_tensor(A)
    M[i, :] = j_scale * M[i, :] + k_scale * M[j, :]
    return M

def rref(A, eps: float = 1e-12):
    """
    Reduced Row Echelon Form via Gauss–Jordan elimination with partial pivoting.

    Algorithm (brief):
      - For each column from left to right, find a pivot row at/below current row r
        with maximal |entry|.
      - Swap it into row r.
      - Scale row r to make the pivot exactly 1.
      - Eliminate that column in all other rows (above and below) to 0.
      - Move to the next row/column until done.
    """
    M = _as_float_tensor(A)
    rows, cols = M.shape
    r = 0                       # next pivot row
    for c in range(cols):
        if r >= rows:
            break

        # Find pivot row (max |entry|) among rows r..rows-1 in column c
        pivot_row = None
        max_val = eps
        for rr in range(r, rows):
            val = abs(M[rr, c].item())
            if val > max_val:
                max_val = val
                pivot_row = rr
        if pivot_row is None:
            continue  # no pivot in this column

        # Move pivot to row r
        if pivot_row != r:
            M = rowswap(M, r, pivot_row)

        # Scale pivot row so pivot becomes 1
        pivot = M[r, c].item()
        if abs(pivot) > eps:
            M = rowscale(M, r, 1.0 / pivot)

        # Eliminate this column in all other rows
        for rr in range(rows):
            if rr == r:
                continue
            factor = M[rr, c].item()
            if abs(factor) > eps:
                M = rowreplacement(M, rr, r, j_scale=1.0, k_scale=-factor)

        r += 1
        if r == rows:
            break

    # Clean tiny roundoff
    M[abs(M) < eps] = 0.0
    return M

if __name__ == "__main__":
    M = torch.tensor([[0, 3, -6, 6, 4], [3, -7, 8, -5, 8], [3, -9, 12, -9, 6]], dtype = torch.float32)

    M = (M, 1, 3.0)
    print(M)
