import unittest
from pathlib import Path

from src.model import load_dataset, train_model, predict_message, evaluate_model
from src.security import inspect_risk
from src.history import save_scan, read_history, clear_history


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.data_path = Path("data/messages.csv")
        self.data = load_dataset(self.data_path)
        self.model, self.vectorizer, self.X_test, self.y_test = train_model(self.data)

    def test_dataset_loads(self):
        self.assertGreaterEqual(len(self.data), 2)
        self.assertEqual(set(self.data["label"]), {"Ham", "Spam"})

    def test_prediction(self):
        label, score = predict_message("Congratulations you won a prize", self.model, self.vectorizer)
        self.assertIn(label, {"Ham", "Spam"})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_evaluation(self):
        metrics, matrix = evaluate_model(self.model, self.vectorizer, self.X_test, self.y_test)
        self.assertIn("accuracy", metrics)
        self.assertEqual(matrix.shape, (2, 2))

    def test_risk(self):
        level, flags = inspect_risk("urgent verify your account password")
        self.assertIn(level, {"High", "Review", "No configured flags"})
        self.assertTrue(flags)

    def test_history_roundtrip(self):
        clear_history()
        save_scan("test message", "Ham", 0.5, "No configured flags", [])
        history = read_history()
        self.assertEqual(len(history), 1)
        clear_history()


if __name__ == "__main__":
    unittest.main()
