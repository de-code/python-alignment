import logging

import numpy as np

from .sequencealigner import SimpleScoring
from .AlignmentMatrix import compute_alignment_matrix

DEFAULT_MATCH_SCORE = 2
DEFAULT_MISMATCH_SCORE = -1
DEFAULT_GAP_SCORE = -3

DEFAULT_SCORING = SimpleScoring(DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE)

def get_logger():
  return logging.getLogger(__name__)

def assert_matrix_equal(m1, m2):
    def to_str(m):
        return '\n'.join([' '.join([str(x) for x in row]) for row in m])
    
    assert to_str(m1) == to_str(m2)

class TestComputeAlignmentMatrix(object):
    def test_equal_sequences(self):
        first = 'abc'
        second = 'abc'
        m = compute_alignment_matrix(first, second, DEFAULT_SCORING, DEFAULT_GAP_SCORE)
        get_logger().info('m:\n%s', m)
        assert_matrix_equal(m, np.array([
            [0, 0, 0, 0],
            [0, DEFAULT_MATCH_SCORE, 0, 0],
            [0, 0, 2 * DEFAULT_MATCH_SCORE, 2 * DEFAULT_MATCH_SCORE + DEFAULT_GAP_SCORE],
            [0, 0, 2 * DEFAULT_MATCH_SCORE + DEFAULT_GAP_SCORE, 3 * DEFAULT_MATCH_SCORE]
        ]))
