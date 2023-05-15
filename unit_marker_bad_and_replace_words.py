import unittest
import marker_bad_and_replace_words


class MarkerBadWordsTestCase(unittest.TestCase):

    def test_openfile(self):
        res = marker_bad_and_replace_words.MarkerBadWords("parasites.txt", "Как бы не так")
        res.list_of_bad_words()
        self.assertEqual(res.bad_words_list, ["таким образом", "как бы"])

    def test_expletive(self):
        res = marker_bad_and_replace_words.MarkerBadWords("parasites.txt", "Как бы не так")
        res.list_of_bad_words()
        self.assertEqual(res.bad_words(), [0, 1])


class SimilarityMarkerTestCase(unittest.TestCase):

    def test_marker_not_in_std(self):
        res = marker_bad_and_replace_words.SimilarityMarker("Как бы сделать", "Сделать хоть как-то").marker_not_in_std()
        self.assertEqual(res, {"как-то", "хоть"})

    def test_marker_not_in_rep(self):
        res = marker_bad_and_replace_words.SimilarityMarker("Как бы сделать", "Сделать хоть как-то").marker_not_in_rep()
        self.assertEqual(res, {"бы", "как"})


