from cpython cimport array

import numpy as np
cimport numpy as np

cdef int imax4(int a, int b, int c, int d):
  if a >= b:
    if a >= c:
      if a >= d:
        return a
      else:
        return d
    else:
      if c >= d:
        return c
      else:
        return d
  else:
    if b >= c:
      if b >= d:
        return b
      else:
        return d
    else:
      if c >= d:
        return c
      else:
        return d


def compute_alignment_matrix(first, second, scoring, int gap_score):
    cdef int m = len(first) + 1
    cdef int n = len(second) + 1
    cdef np.ndarray[np.int_t, ndim=2] f = np.empty((m, n), int)
    f[:, 0] = 0
    f[0, :] = 0
    for i in range(1, m):
        for j in range(1, n):
            f[i, j] = imax4(
                0,

                # Match elements.
                f[i - 1, j - 1] \
                    + scoring(first[i - 1], second[j - 1]),

                # Gap on sequenceA.
                f[i, j - 1] + gap_score,

                # Gap on sequenceB.
                f[i - 1, j] + gap_score
            )
    return f

def compute_alignment_matrix_simple_scoring(first, second, int match_score, int mismatch_score, int gap_score):
    cdef list cfirst = list(first)
    cdef list csecond = list(second)
    cdef int m = len(cfirst) + 1
    cdef int n = len(csecond) + 1
    cdef np.ndarray[np.int_t, ndim=2] f = np.empty((m, n), int)
    f[:, 0] = 0
    f[0, :] = 0
    for i in range(1, m):
        for j in range(1, n):
            f[i, j] = imax4(
                0,

                # Match elements.
                f[i - 1, j - 1] \
                    + (match_score if cfirst[i - 1] == csecond[j - 1] else mismatch_score),

                # Gap on sequenceA.
                f[i, j - 1] + gap_score,

                # Gap on sequenceB.
                f[i - 1, j] + gap_score
            )
    return f

def compute_alignment_matrix_simple_scoring_int(
    np.ndarray[np.int32_t] first, np.ndarray[np.int32_t] second,
    int match_score, int mismatch_score, int gap_score):

    cdef int[:] cfirst = first.astype(np.int32)
    cdef int[:] csecond = second.astype(np.int32)

    cdef int m = len(cfirst) + 1
    cdef int n = len(csecond) + 1
    # cdef np.ndarray[np.int32_t, ndim=2] f = np.empty((m, n), dtype=np.int32)
    cdef np.ndarray[np.int_t, ndim=2] f = np.empty((m, n), dtype=np.int)
    f[:, 0] = 0
    f[0, :] = 0
    for i in range(1, m):
        for j in range(1, n):
            f[i, j] = imax4(
                0,

                # Match elements.
                f[i - 1, j - 1] \
                    + (match_score if cfirst[i - 1] == csecond[j - 1] else mismatch_score),

                # Gap on sequenceA.
                f[i, j - 1] + gap_score,

                # Gap on sequenceB.
                f[i - 1, j] + gap_score
            )
    return f

def compute_alignment_matrix_simple_scoring_str(
    unicode[:] cfirst, unicode[:] csecond,
    int match_score, int mismatch_score, int gap_score):

    cdef int m = len(cfirst) + 1
    cdef int n = len(csecond) + 1
    cdef np.ndarray[np.int_t, ndim=2] f = np.empty((m, n), int)
    f[:, 0] = 0
    f[0, :] = 0
    for i in range(1, m):
        for j in range(1, n):
            f[i, j] = max(
                0,

                # Match elements.
                f[i - 1, j - 1] \
                    + (match_score if cfirst[i - 1] == csecond[j - 1] else mismatch_score),

                # Gap on sequenceA.
                f[i, j - 1] + gap_score,

                # Gap on sequenceB.
                f[i - 1, j] + gap_score
            )
    return f
