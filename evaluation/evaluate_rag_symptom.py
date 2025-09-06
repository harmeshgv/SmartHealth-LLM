import os
import sys  
import time
import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm

# Add project root to sys.path BEFORE any other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import TEST_CASES_CSV
from backend.utils.symtom_checker import DiseaseMatcherAgent


class TopKEvaluator:
    def __init__(self, k=3, csv_path=TEST_CASES_CSV):
        self.k = k
        self.matcher = DiseaseMatcherAgent(top_k=self.k)  # pass k on init for default retriever top_k
        self.df = pd.read_csv(csv_path)
        self.y_true = []
        self.y_pred_topk = []
        self.response_times = []

    def process(self):
        print(f"🔍 Matching symptoms to top-{self.k} diseases...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            symptoms = row["Symptoms"]
            expected = row["disease"]
            start_time = time.time()
            
            # Get top-k matches (list of disease strings)
            predicted = self.matcher.match(symptoms, top_k=self.k)
            if predicted is None:
                predicted = []

            end_time = time.time()
            self.response_times.append((end_time - start_time) * 1000)  # ms

            self.y_true.append(expected)
            self.y_pred_topk.append(predicted)

    def evaluate(self):
        correct_topk = 0
        total = len(self.y_true)

        for true, pred_list in zip(self.y_true, self.y_pred_topk):
            true_lower = str(true).strip().lower()
            pred_list_lower = [str(p).strip().lower() for p in pred_list]

            if true_lower in pred_list_lower:
                correct_topk += 1

        accuracy = correct_topk / total if total > 0 else 0
        average_response_time = np.mean(self.response_times)

        print(f"\n🎯 Top-{self.k} Evaluation Results:")
        print(f"✔️ Total cases: {total}")
        print(f"✅ Correct within Top-{self.k}: {correct_topk}")
        print(f"📊 Top-{self.k} Accuracy: {accuracy * 100:.4f} %")
        print(f"⚡ Avg Response Time: {average_response_time:.2f} ms")    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Top-K Accuracy")
    parser.add_argument("k", type=int, nargs="?",  help="Top-K value to evaluate; if omitted, runs k=1,3,5")
    args = parser.parse_args()

    if args.k is None:
        for i in range(1, 6, 2):  # k = 1, 3, 5
            evaluator = TopKEvaluator(k=i)
            evaluator.process()
            evaluator.evaluate()
            print("-" * 40)
    else:
        evaluator = TopKEvaluator(k=args.k)
        evaluator.proc
