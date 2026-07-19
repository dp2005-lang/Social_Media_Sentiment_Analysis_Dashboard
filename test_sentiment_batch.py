import unittest

import pandas as pd

from sentiment_batch import find_text_column, predict_dataframe


class RecordingVectorizer:
    def __init__(self):
        self.values = None

    def transform(self, values):
        self.values = values
        return values


class RecordingModel:
    def __init__(self):
        self.values = None

    def predict(self, values):
        self.values = values
        return ["neutral"] * len(values)


class SentimentBatchTests(unittest.TestCase):
    def test_find_text_column_normalizes_case_and_spaces(self):
        dataframe = pd.DataFrame({" Full Text ": ["Example"]})

        self.assertEqual(find_text_column(dataframe), " Full Text ")

    def test_predict_dataframe_handles_missing_and_numeric_values_in_one_batch(self):
        dataframe = pd.DataFrame({"tweet": ["Hello", None, 42]})
        model = RecordingModel()
        vectorizer = RecordingVectorizer()

        result = predict_dataframe(
            dataframe,
            model=model,
            vectorizer=vectorizer,
            preprocess=lambda value: value.lower(),
        )

        self.assertEqual(vectorizer.values, ["hello", "", "42"])
        self.assertEqual(model.values, ["hello", "", "42"])
        self.assertEqual(result["prediction"].tolist(), ["neutral"] * 3)

    def test_predict_dataframe_accepts_a_header_only_csv(self):
        dataframe = pd.DataFrame(columns=["text"])
        model = RecordingModel()
        vectorizer = RecordingVectorizer()

        result = predict_dataframe(
            dataframe,
            model=model,
            vectorizer=vectorizer,
            preprocess=str.lower,
        )

        self.assertEqual(result.columns.tolist(), ["text", "prediction"])
        self.assertTrue(result.empty)
        self.assertIsNone(vectorizer.values)
        self.assertIsNone(model.values)

    def test_predict_dataframe_rejects_unsupported_columns(self):
        dataframe = pd.DataFrame({"created_at": ["2026-07-20"]})

        with self.assertRaisesRegex(ValueError, "CSV needs a text column"):
            predict_dataframe(
                dataframe,
                model=RecordingModel(),
                vectorizer=RecordingVectorizer(),
                preprocess=str.lower,
            )


if __name__ == "__main__":
    unittest.main()
