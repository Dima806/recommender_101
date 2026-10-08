"""Cosine similarity, from scratch, kept to the top-k neighbours per row.

This is the heart of neighbourhood collaborative filtering: "similar" is literally
cosine similarity between rating vectors. We L2-normalize each row, then a dot
product *is* the cosine. We keep only the strongest ``k`` neighbours per row (as a
sparse matrix) so the memory cost stays linear in items, not quadratic.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize


def topk_cosine(matrix: csr_matrix, k: int, block_size: int = 512) -> csr_matrix:
    """Top-k cosine-similarity neighbours per row, returned as a sparse (n, n) matrix.

    Self-similarity is removed so a row is never its own neighbour. Computed in
    row blocks to bound peak memory.
    """
    normed = normalize(matrix.astype(np.float64), norm="l2", axis=1).tocsr()
    n_rows = normed.shape[0]
    n_neighbors = min(k, max(n_rows - 1, 1))

    blocks = []
    for start in range(0, n_rows, block_size):
        end = min(start + block_size, n_rows)
        sims = (normed[start:end] @ normed.T).toarray()
        for local, global_row in enumerate(range(start, end)):
            sims[local, global_row] = 0.0  # drop self-similarity

        top = np.argpartition(-sims, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        rows = np.repeat(np.arange(end - start), n_neighbors)
        cols = top.ravel()
        local_rows = np.arange(end - start)[:, None]
        vals = sims[local_rows, top].ravel()
        blocks.append(csr_matrix((vals, (rows, cols)), shape=(end - start, n_rows)))

    return vstack(blocks).tocsr()
