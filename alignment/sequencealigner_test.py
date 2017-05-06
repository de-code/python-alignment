"""
Unit test
"""
from .vocabulary import Vocabulary

from .sequence import Sequence

from .sequencealigner import (
    GlobalSequenceAligner,
    StrictGlobalSequenceAligner,
    LocalSequenceAligner,
    SmithWatermanAligner,
    SimpleScoring
)

DEFAULT_MATCH_SCORE = 3
DEFAULT_MISMATCH_SCORE = -1
DEFAULT_GAP_SCORE = -2

def _align(
        first,
        second,
        aligner):
    vocab = Vocabulary()
    score, encodeds = aligner.align(
        vocab.encodeSequence(Sequence(first)),
        vocab.encodeSequence(Sequence(second)),
        backtrace=True
    )
    return score, [vocab.decodeSequenceAlignment(encoded) for encoded in encodeds]

def _global_align(
        first, second,
        match_score=DEFAULT_MATCH_SCORE,
        mismatch_score=DEFAULT_MISMATCH_SCORE,
        gap_score=DEFAULT_GAP_SCORE):
    scoring = SimpleScoring(match_score, mismatch_score)
    return _align(first, second, aligner=GlobalSequenceAligner(scoring, gap_score))

def _strict_global_align(
        first, second,
        match_score=DEFAULT_MATCH_SCORE,
        mismatch_score=DEFAULT_MISMATCH_SCORE,
        gap_score=DEFAULT_GAP_SCORE):
    scoring = SimpleScoring(match_score, mismatch_score)
    return _align(first, second, aligner=StrictGlobalSequenceAligner(scoring, gap_score))

def _local_align(
        first, second,
        match_score=DEFAULT_MATCH_SCORE,
        mismatch_score=DEFAULT_MISMATCH_SCORE,
        gap_score=DEFAULT_GAP_SCORE):
    scoring = SimpleScoring(match_score, mismatch_score)
    return _align(first, second, aligner=LocalSequenceAligner(scoring, gap_score))


class CommonSequenceAlignerTests(object):
    def _align(self, _first, _second):
        raise Exception('not implemented')

    # def test_no_match(self):
    #     score, alignments = self._align('ab', 'xy')
    #     assert len(alignments) == 0
    #     assert score == 0

    def test_exact_match(self):
        score, alignments = self._align('ab', 'ab')
        print("alignment:", alignments[0])
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a b'
        assert str(alignments[0].second) == 'a b'
        assert alignments[0].percentIdentity() == 100.0
        assert alignments[0].percentSimilarity() == 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 2
        assert alignments[0].score == score

    def test_exact_left_partial_match(self):
        score, alignments = self._align('xaby', 'ab')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a b'
        assert str(alignments[0].second) == 'a b'
        assert alignments[0].score == score
        assert alignments[0].percentIdentity() == 100.0
        assert alignments[0].percentSimilarity() == 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 2

    def test_exact_right_partial_match(self):
        score, alignments = self._align('ab', 'xaby')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a b'
        assert str(alignments[0].second) == 'a b'
        assert alignments[0].percentIdentity() == 100.0
        assert alignments[0].percentSimilarity() == 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 2
        assert alignments[0].score == score

    def test_exact_left_partial_match_with_gap(self):
        score, alignments = self._align('xaby', 'aob')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a - b'
        assert str(alignments[0].second) == 'a o b'
        assert alignments[0].percentIdentity() == 2 / 3 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 3 * 100.0
        assert alignments[0].percentGap() == 1 / 3 * 100.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_GAP_SCORE
        assert alignments[0].score == score

    def test_exact_left_partial_match_with_mismatch(self):
        score, alignments = self._align('xamby', 'aob')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a m b'
        assert str(alignments[0].second) == 'a o b'
        assert alignments[0].percentIdentity() == 2 / 3 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 3 * 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_MISMATCH_SCORE
        assert alignments[0].score == score


class TestGlobalSequenceAligner(CommonSequenceAlignerTests):
    def _align(self, first, second):
        return _global_align(first, second)

    def test_multiple_alignments(self):
        score, alignments = self._align('xabcabcy', 'abc')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a b c'
        assert str(alignments[0].second) == 'a b c'
        assert alignments[0].percentIdentity() == 3 / 3 * 100.0
        assert alignments[0].percentSimilarity() == 3 / 3 * 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 3
        assert alignments[0].score == score

    def test_shortest_path_alignment(self):
        # this tests that it doesn't pick longer paths on the way
        # (e.g. goes up instead of diagonally)
        score, alignments = self._align('aac', 'bac')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a a c'
        assert str(alignments[0].second) == 'b a c'
        assert alignments[0].percentIdentity() == 2 / 3 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 3 * 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_MISMATCH_SCORE
        assert alignments[0].score == score


class TestStrictGlobalSequenceAligner(CommonSequenceAlignerTests):
    def _align(self, first, second):
        return _strict_global_align(first, second)

    def test_exact_left_partial_match(self):
        score, alignments = self._align('xaby', 'ab')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'x a b y'
        assert str(alignments[0].second) == '- a b -'
        assert alignments[0].score == score
        assert alignments[0].percentIdentity() == 2 / 4 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 4 * 100.0
        assert alignments[0].percentGap() == 2 / 4 * 100.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_GAP_SCORE * 2

    def test_exact_right_partial_match(self):
        score, alignments = self._align('ab', 'xaby')
        assert len(alignments) == 1
        assert str(alignments[0].first) == '- a b -'
        assert str(alignments[0].second) == 'x a b y'
        assert alignments[0].percentIdentity() == 2 / 4 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 4 * 100.0
        assert alignments[0].percentGap() == 2 / 4 * 100.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_GAP_SCORE * 2
        assert alignments[0].score == score

    def test_exact_left_partial_match_with_gap(self):
        score, alignments = self._align('xaby', 'aob')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'x a - b y'
        assert str(alignments[0].second) == '- a o b -'
        assert alignments[0].percentIdentity() == 2 / 5 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 5 * 100.0
        assert alignments[0].percentGap() == 3 / 5 * 100.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_GAP_SCORE * 3
        assert alignments[0].score == score

    def test_exact_left_partial_match_with_mismatch(self):
        score, alignments = self._align('xamby', 'aob')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'x a m b y'
        assert str(alignments[0].second) == '- a o b -'
        assert alignments[0].percentIdentity() == 2 / 5 * 100.0
        assert alignments[0].percentSimilarity() == 2 / 5 * 100.0
        assert alignments[0].percentGap() == 2 / 5 * 100.0
        assert score == DEFAULT_MATCH_SCORE * 2 + DEFAULT_MISMATCH_SCORE + DEFAULT_GAP_SCORE * 2
        assert alignments[0].score == score

    def test_multiple_alignments(self):
        score, alignments = self._align('xabcabcy', 'abc')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'x a b c a b c y'
        assert str(alignments[0].second) == '- a b c - - - -'
        assert alignments[0].percentIdentity() == 3 / 8 * 100.0
        assert alignments[0].percentSimilarity() == 3 / 8 * 100.0
        assert alignments[0].percentGap() == 5 / 8 * 100.0
        assert score == DEFAULT_MATCH_SCORE * 3 + DEFAULT_GAP_SCORE * 5
        assert alignments[0].score == score


class TestLocalSequenceAligner(CommonSequenceAlignerTests):
    def _align(self, first, second):
        return _local_align(first, second)

    def test_multiple_alignments(self):
        score, alignments = self._align('xabcabcy', 'abc')
        assert len(alignments) == 2
        assert str(alignments[0].first) == 'a b c'
        assert str(alignments[0].second) == 'a b c'
        assert alignments[0].percentIdentity() == 3 / 3 * 100.0
        assert alignments[0].percentSimilarity() == 3 / 3 * 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 3
        assert alignments[0].score == score

    def test_shortest_path_alignment(self):
        # this tests that it doesn't pick longer paths on the way
        # (e.g. goes up instead of diagonally)
        score, alignments = self._align('aac', 'bac')
        assert len(alignments) == 1
        assert str(alignments[0].first) == 'a c'
        assert str(alignments[0].second) == 'a c'
        assert alignments[0].percentIdentity() == 3 / 3 * 100.0
        assert alignments[0].percentSimilarity() == 3 / 3 * 100.0
        assert alignments[0].percentGap() == 0.0
        assert score == DEFAULT_MATCH_SCORE * 2
        assert alignments[0].score == score

class TestSmithWatermannAligner(TestLocalSequenceAligner):
    def _align(self, first, second):
        scoring = SimpleScoring(DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE)
        return _align(first, second, aligner=SmithWatermanAligner(scoring, DEFAULT_GAP_SCORE))

class TestSmithWatermannAlignerWithString(TestSmithWatermannAligner):
    def _align(self, first, second):
        scoring = SimpleScoring(DEFAULT_MATCH_SCORE, DEFAULT_MISMATCH_SCORE)
        aligner = SmithWatermanAligner(scoring, DEFAULT_GAP_SCORE)
        return aligner.align(
            Sequence(first),
            Sequence(second),
            backtrace=True,
            gap='-'
        )
