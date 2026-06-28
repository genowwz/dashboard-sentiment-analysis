import importlib
import unittest
from unittest.mock import patch


streamlit_app = importlib.import_module("streamlit_app")


class WordcloudSourceTests(unittest.TestCase):
    def test_prediction_mode_uses_uploaded_words_for_wordcloud(self):
        latest = {
            "mode": "prediction",
            "words_pos": {"upload": 5},
            "words_neg": {"buruk": 2},
        }

        with patch.object(streamlit_app, "extract_words_from_testset", return_value=({"test": 1}, {"test_neg": 1})):
            result = streamlit_app.get_wordcloud_data_for_display(latest)

        self.assertEqual(result, ({"upload": 5}, {"buruk": 2}))

    def test_evaluation_mode_falls_back_to_testset(self):
        latest = {"mode": "evaluation"}

        with patch.object(streamlit_app, "extract_words_from_testset", return_value=({"test": 1}, {"test_neg": 1})):
            result = streamlit_app.get_wordcloud_data_for_display(latest)

        self.assertEqual(result, ({"test": 1}, {"test_neg": 1}))


if __name__ == "__main__":
    unittest.main()
