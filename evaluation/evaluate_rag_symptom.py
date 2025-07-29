import os
import sys  
import time
import numpy as np
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
        self.response_times = []

    def process(self):
        print(f"🔍 Matching symptoms to top-{self.k} diseases...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            symptoms = row["Symptoms"]
            expected = row["disease"]
            start_time = time.time()
            matches = self.matcher.match(symptoms, top_k=self.k)
            predicted = [match[0] for match in matches] if matches else []
            end_time = time.time()
            self.response_times.append((end_time - start_time) * 1000)

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
        average_response_time = np.mean(self.response_times)

        print(f"\n🎯 Top-{self.k} Evaluation Results:")
        print(f"✔️ Total cases: {total}")
        print(f"✅ Correct within Top-{self.k}: {correct_topk}")
        print(f"📊 Top-{self.k} Accuracy: {accuracy * 100:.8f} %")
        print(f"Speed accuracy response time : {average_response_time}" )    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Top-K Accuracy")
    parser.add_argument("k", type=int, nargs="?",  help="Top-K value to evaluate (leave empty for full evaluation 1, 3, 5)")
    args = parser.parse_args()
    if args.k == None:
        for i in range(1,6,2):
                evaluator = TopKEvaluator(k=i)
                evaluator.process()
                evaluator.evaluate()
                print("-" * 40)

            
    else:
        evaluator = TopKEvaluator(k=args.k)
        evaluator.process()
        evaluator.evaluate()
