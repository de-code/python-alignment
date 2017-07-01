from __future__ import absolute_import, print_function

import timeit

import numpy as np

from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring
from alignment.AlignmentMatrix import (
  compute_alignment_matrix,
  compute_alignment_matrix_simple_scoring,
  compute_alignment_matrix_simple_scoring_int
)

DEFAULT_MATCH_SCORE = 2
DEFAULT_MISMATCH_SCORE = -1
DEFAULT_GAP_SCORE = -3

DEFAULT_SCORING = SimpleScoring(DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE)

SHORT_STRING1 = 'abc'
SHORT_STRING2 = 'def'

LONG_STRING1 = 'abcefghijk' * 100
LONG_STRING2 = list(reversed(LONG_STRING1))

v = Vocabulary()

def encode_str(s):
    return np.array([int(v.encode(x)) for x in s], dtype=np.int32)

LONG_ENCODED1 = encode_str(LONG_STRING1)
LONG_ENCODED2 = encode_str(LONG_STRING2)

def compute_alignment_matrix_py(first, second, scoring, gap_score):
    m = len(first) + 1
    n = len(second) + 1
    f = np.empty((m, n), int)
    f[:, 0] = 0
    f[0, :] = 0
    for i in range(1, m):
        for j in range(1, n):
            f[i, j] = max(
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

def test_compute_alignment_matrix_scoring_fn_py():
    compute_alignment_matrix_py(LONG_STRING1, LONG_STRING2, DEFAULT_SCORING, DEFAULT_GAP_SCORE)

def test_compute_alignment_matrix_scoring_fn():
    compute_alignment_matrix(LONG_STRING1, LONG_STRING2, DEFAULT_SCORING, DEFAULT_GAP_SCORE)

def test_compute_alignment_matrix_simple_scoring():
    compute_alignment_matrix_simple_scoring(
        LONG_STRING1, LONG_STRING2,
        DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE, DEFAULT_GAP_SCORE
    )

def test_compute_alignment_matrix_simple_scoring_int():
    compute_alignment_matrix_simple_scoring_int(
        LONG_ENCODED1, LONG_ENCODED2,
        DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE, DEFAULT_GAP_SCORE
    )

# def test_compute_alignment_matrix_simple_scoring_str():
#     compute_alignment_matrix_simple_scoring_str(
#         LONG_STRING1, LONG_STRING2,
#         DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE, DEFAULT_GAP_SCORE
#     )

def report_timing(fn, number=1):
    timeit_result_ms = timeit.timeit(
        fn + "()",
        setup="from __main__ import " + fn,
        number=number
    ) * 1000
    print("{} ({}x):\n{:f} ms / it ({:f} ms total)\n".format(
        fn,
        number,
        timeit_result_ms / number,
        timeit_result_ms 
    ))

def main():
    print("len LONG_STRING1: {}\n".format(len(LONG_STRING1)))
    print("len LONG_ENCODED1: {}\n".format(len(LONG_ENCODED1)))
    report_timing("test_compute_alignment_matrix_scoring_fn_py")
    report_timing("test_compute_alignment_matrix_scoring_fn")
    report_timing("test_compute_alignment_matrix_simple_scoring")
    report_timing("test_compute_alignment_matrix_simple_scoring_int")

if __name__ == "__main__":
    main()
