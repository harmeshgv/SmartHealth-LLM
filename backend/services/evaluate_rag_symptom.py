import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Allow importing from parent folders
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.services.symptom_to_disease import DiseaseMatcher
class RagEvaluator:
    def __init__(self, csv_path="backend/data/test_symptom_cases.csv"):
        self.matcher = DiseaseMatcher(vectorstore_path="backend/Vector/symptom_faiss_db")
        self.df = pd.read_csv(csv_path)
        self.y_true = []
        self.y_pred = []

    def process(self):
        print("🔍 Matching symptoms to diseases...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing"):
            symptom_text = row["symptoms"]
            expected = row["expected_disease"]

            # Get top-1 prediction from vectorstore
            matches = self.matcher.match(symptom_text, top_k=1)
            predicted = matches[0][0] if matches else "Unknown"

            self.y_true.append(expected)
            self.y_pred.append(predicted)

    def evaluate(self):
        correct = 0
        total = len(self.y_true)

        for true, pred in zip(self.y_true, self.y_pred):
            if true.strip().lower() == pred.strip().lower():
                correct += 1

        accuracy = correct / total if total > 0 else 0

        print("\n✅ Evaluation Results:")
        print(f"✔️ Total: {total}")
        print(f"🎯 Correct: {correct}")
        print(f"📊 Accuracy: {accuracy:.2f}")

    # Optional: save to file
        with open("evaluation/rag_model_score.txt", "w") as f:
            f.write(f"Total: {total}\n")
            f.write(f"Correct: {correct}\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")


if __name__ == "__main__":
    evaluator = RagEvaluator()
    evaluator.process()
    evaluator.evaluate()
