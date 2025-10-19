"""Utilities for solving word analogies using NumPy vectors."""
from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Union, List

import numpy as np


VectorLike = Union[Sequence[float], np.ndarray]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row-wise L2 normalized copy of ``matrix``.

    Rows with zero norm are left untouched to avoid division by zero.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero: keep zero rows unchanged.
    normalized = matrix.copy()
    non_zero_rows = norms[:, 0] > 0
    normalized[non_zero_rows] /= norms[non_zero_rows]
    return normalized


def solve_analogy(
    word_a: str,
    word_b: str,
    word_c: str,
    embeddings: Mapping[str, VectorLike],
    *,
    top_k: int = 1,
) -> Union[str, List[Tuple[str, float]]]:
    """Solve analogies of the form ``a`` is to ``b`` as ``c`` is to ``?``.

    Parameters
    ----------
    word_a, word_b, word_c:
        Words that define the analogy ``a : b :: c : ?``. All three words must
        be present in ``embeddings``.
    embeddings:
        Mapping from words to vector representations. All vectors must share
        the same dimensionality. Any ``Sequence`` convertible to a NumPy array
        is accepted.
    top_k:
        Number of best-scoring candidates to return. When ``top_k`` is ``1``
        (the default), a single word is returned. For larger values, a list of
        ``(word, score)`` tuples ordered by decreasing cosine similarity is
        returned.

    Returns
    -------
    str or list of (str, float)
        The best candidate completing the analogy, or a list of top candidates
        with their cosine similarity scores when ``top_k`` > 1.

    Raises
    ------
    ValueError
        If a word is missing from ``embeddings``, the vectors have inconsistent
        dimensions, or ``top_k`` is less than 1.
    """

    if top_k < 1:
        raise ValueError("top_k must be greater than or equal to 1")

    required_words = (word_a, word_b, word_c)
    for word in required_words:
        if word not in embeddings:
            raise ValueError(f"Word '{word}' not found in the embeddings.")

    words: List[str] = []
    vectors: List[np.ndarray] = []
    vector_length = None

    for word, vector in embeddings.items():
        arr = np.asarray(vector, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("Embedding vectors must be one-dimensional.")
        if vector_length is None:
            vector_length = arr.shape[0]
        elif arr.shape[0] != vector_length:
            raise ValueError("All embedding vectors must share the same length.")
        words.append(word)
        vectors.append(arr)

    matrix = np.vstack(vectors)
    normalized_matrix = _normalize_rows(matrix)

    word_to_index = {word: idx for idx, word in enumerate(words)}
    vec_a = normalized_matrix[word_to_index[word_a]]
    vec_b = normalized_matrix[word_to_index[word_b]]
    vec_c = normalized_matrix[word_to_index[word_c]]

    target_vector = vec_b - vec_a + vec_c
    target_norm = np.linalg.norm(target_vector)
    if target_norm == 0:
        raise ValueError("The analogy target vector has zero length.")
    target_vector /= target_norm

    similarities = normalized_matrix @ target_vector

    excluded_indices = {word_to_index[word] for word in required_words}
    similarities[list(excluded_indices)] = -np.inf

    top_k = min(top_k, len(words) - len(excluded_indices))
    if top_k <= 0:
        raise ValueError("Not enough candidate words to compute the analogy.")

    best_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
    # Sort the selected candidates by similarity in descending order.
    best_indices = best_indices[np.argsort(-similarities[best_indices])]

    if top_k == 1:
        return words[int(best_indices[0])]

    return [(words[int(idx)], float(similarities[int(idx)])) for idx in best_indices]


def most_similar(
    word: str,
    embeddings: Mapping[str, VectorLike],
    *,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Return the ``top_n`` words most similar to ``word`` using cosine similarity.

    Parameters
    ----------
    word:
        Reference word whose neighbours we want to retrieve. It must be present
        in ``embeddings``.
    embeddings:
        Mapping from words to vector representations. All vectors must share the
        same dimensionality. Any ``Sequence`` convertible to a NumPy array is
        accepted.
    top_n:
        Number of similar words to retrieve. Must be at least 1.

    Returns
    -------
    list of (str, float)
        A list with the closest words and their cosine similarity to ``word``
        ordered from most to least similar.

    Raises
    ------
    ValueError
        If ``word`` is missing, the vectors have inconsistent dimensions or
        ``top_n`` is less than 1.
    """

    if top_n < 1:
        raise ValueError("top_n must be greater than or equal to 1")

    if word not in embeddings:
        raise ValueError(f"Word '{word}' not found in the embeddings.")

    words: List[str] = []
    vectors: List[np.ndarray] = []
    vector_length = None

    for current_word, vector in embeddings.items():
        arr = np.asarray(vector, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("Embedding vectors must be one-dimensional.")
        if vector_length is None:
            vector_length = arr.shape[0]
        elif arr.shape[0] != vector_length:
            raise ValueError("All embedding vectors must share the same length.")
        words.append(current_word)
        vectors.append(arr)

    matrix = np.vstack(vectors)
    normalized_matrix = _normalize_rows(matrix)

    word_to_index = {current_word: idx for idx, current_word in enumerate(words)}
    reference_vector = normalized_matrix[word_to_index[word]]

    similarities = normalized_matrix @ reference_vector

    # Exclude the reference word from the results by setting its similarity to
    # negative infinity.
    similarities[word_to_index[word]] = -np.inf

    available_candidates = len(words) - 1
    top_n = min(top_n, available_candidates)
    if top_n <= 0:
        raise ValueError("Not enough candidate words to compute similarities.")

    best_indices = np.argpartition(-similarities, top_n - 1)[:top_n]
    best_indices = best_indices[np.argsort(-similarities[best_indices])]

    return [(words[int(idx)], float(similarities[int(idx)])) for idx in best_indices]


def find_most_similar(
    word: str,
    embeddings: Mapping[str, VectorLike],
    *,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Backward compatible wrapper for :func:`most_similar`.

    Historically the helper that exposes the behaviour of :func:`most_similar`
    was named ``find_most_similar`` in course materials. Some notebooks still
    import or call it with that name.  To keep those references working we
    expose a thin wrapper that simply forwards the call to
    :func:`most_similar`.

    Parameters
    ----------
    word, embeddings, top_n:
        Passed directly to :func:`most_similar`.
    """

    return most_similar(word, embeddings, top_n=top_n)


__all__ = ["solve_analogy", "most_similar", "find_most_similar"]
