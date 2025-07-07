import os
import sys  

# Add project root to sys.path BEFORE any other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import argparse

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from backend.config import TEST_CASES_CSV, VECTOR_DIR
from backend.services.symptom_to_disease import DiseaseMatcher


class TopKEvaluator:
    def __init__(self, k=3, csv_path=TEST_CASES_CSV):
        self.k = k
        self.matcher = DiseaseMatcher(vectorstore_path=VECTOR_DIR)
        self.df = pd.read_csv(csv_path)
        self.y_true = []
        self.y_pred_topk = []

    def process(self):
        print(f"🔍 Matching symptoms to top-{self.k} diseases...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            symptoms = row["symptoms"]
            expected = row["expected_disease"]

            matches = self.matcher.match(symptoms, top_k=self.k)
            predicted = [match[0] for match in matches] if matches else []

            self.y_true.append(expected)
            self.y_pred_topk.append(predicted)

    def evaluate(self):
        correct_topk = 0
        total = len(self.y_true)

        for true, pred_list in zip(self.y_true, self.y_pred_topk):
            true_lower = true.strip().lower()
            pred_list_lower = [p.strip().lower() for p in pred_list]

            if true_lower in pred_list_lower:
                correct_topk += 1

        accuracy = correct_topk / total if total > 0 else 0

        print(f"\n🎯 Top-{self.k} Evaluation Results:")
        print(f"✔️ Total cases: {total}")
        print(f"✅ Correct within Top-{self.k}: {correct_topk}")
        print(f"📊 Top-{self.k} Accuracy: {accuracy:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Top-K Accuracy")
    parser.add_argument("k", type=int, nargs="?", default=3, help="Top-K value to evaluate (default=3)")
    args = parser.parse_args()

    evaluator = TopKEvaluator(k=args.k)
    evaluator.process()
    evaluator.evaluate()
