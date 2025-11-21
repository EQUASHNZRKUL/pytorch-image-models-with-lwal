#!/usr/bin/env python3
import sys
import re
import numpy as np
import ast

# --------------------------
# Parse stdin where each row begins with "True XX:"
# --------------------------
# def parse_confusion_matrix_using_true(text):
#     pattern = re.compile(r"True\s*\d{1,2}\s*:", flags=re.IGNORECASE)
#     matches = list(pattern.finditer(text))

#     if not matches:
#         raise ValueError("No 'True XX:' markers found in input.")

#     rows = []
#     for i, match in enumerate(matches):
#         start = match.end()
#         end = matches[i+1].start() if i+1 < len(matches) else len(text)
#         chunk = text[start:end]

#         nums = re.findall(r"[-+]?\d*\.\d+|\d+", chunk)
#         if not nums:
#             raise ValueError(f"No numbers found after {match.group(0)}")
#         rows.append([float(x) for x in nums])

#     mat = np.array(rows, dtype=float)
#     lengths = [len(r) for r in rows]
#     if len(set(lengths)) != 1:
#         raise ValueError(f"Inconsistent row lengths found: {lengths}")

#     return mat

def parse_python_matrix(text):
    obj = ast.literal_eval(text)
    arr = np.array(obj, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Parsed object is not 2-dimensional.")
    return 1-arr

# --------------------------
# Heterogeneity metrics
# --------------------------
def shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    s = np.sum(p)
    if s == 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def gini_coefficient(x):
    x = np.asarray(x, dtype=float).flatten()
    if np.allclose(x, 0):
        return 0.0
    n = x.size
    mean = np.mean(x)
    if mean == 0:
        return 0.0
    diff_sum = np.sum(np.abs(x[:, None] - x[None, :]))
    return diff_sum / (2.0 * n * n * mean)


def coefficient_of_variation(x):
    x = np.asarray(x, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x)
    return sigma / mu if mu != 0 else float("nan")


def simpson_index(p):
    p = np.asarray(p, dtype=float)
    s = np.sum(p)
    if s == 0:
        return 0.0
    p = p / s
    return 1.0 - np.sum(p * p)


# --------------------------
# Main
# --------------------------
def main():
    print("Paste your confusion matrix (Ctrl-D to finish):\n")
    text = sys.stdin.read()

    mat = parse_python_matrix(text)

    n = mat.shape[0]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square for off-diagonal extraction.")

    print(f"\nParsed matrix: {n} x {n}")
    print(mat)

    # ------------------------------
    # Extract upper-triangle off-diagonal entries
    # ------------------------------
    # (i < j)
    offdiag = mat[np.triu_indices(n, k=1)]

    print(f"\nNumber of off-diagonal values: {offdiag.size}")
    print(offdiag)

    # ------------------------------
    # Compute heterogeneity on off-diagonal 1-D vector
    # ------------------------------
    shannon_val  = shannon_entropy(offdiag)
    gini_val     = gini_coefficient(offdiag)
    cv_val       = coefficient_of_variation(offdiag)
    simpson_val  = simpson_index(offdiag)

    # ------------------------------
    # Output results
    # ------------------------------
    print("\n=== Heterogeneity Metrics (Off-Diagonal Only) ===")
    print(f"Shannon Entropy:       {shannon_val}")
    print(f"Gini Coefficient:      {gini_val}")
    print(f"Coefficient Variation: {cv_val}")
    print(f"Simpson Index:         {simpson_val}")
    print(f"{shannon_val}, {gini_val}, {cv_val}, {simpson_val}")


if __name__ == "__main__":
    main()
