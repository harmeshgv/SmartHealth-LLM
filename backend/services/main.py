import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Import the disease matcher agent from your backend utils
from backend.utils.DIseaseMatcherAgent import DiseaseMatcherAgent

if __name__ == "__main__":
    # Initialize the matcher with the path to your FAISS vector DB and disease metadata
    matcher = DiseaseMatcherAgent()

    # Input symptoms (word-based matching expected)
    user_input = """Feelings of a fast, fluttering or pounding heartbeat, called palpitations.
Chest pain.
Dizziness.
Fatigue.
Lightheadedness.
Reduced ability to exercise.
Shortness of breath.
Weakness.
Some people with atrial fibrillation (AFib) don't notice any symptoms.
Atrial fibrillation may be:
Occasional, also called paroxysmal atrial fibrillation.AFibsymptoms come and go. The symptoms usually last for a few minutes to hours. Some people have symptoms for as long as a week. The episodes can happen repeatedly. Symptoms might go away on their own. Some people with occasionalAFibneed treatment.
Persistent.The irregular heartbeat is constant. The heart rhythm does not reset on its own. If symptoms occur, medical treatment is needed to correct the heart rhythm.
Long-standing persistent.This type ofAFibis constant and lasts longer than 12 months. Medicines or a procedure are needed to correct the irregular heartbeat.
Permanent.In this type of atrial fibrillation, the irregular heart rhythm can't be reset. Medicines are needed to control the heart rate and to prevent blood clots."""

    # Perform the matching
    matches = matcher.match(user_input, top_k=3)


    # Display the results
    print("\n🔎 Top Matches:")
    if not matches:
        print("No diseases matched.")
    else:
        for disease, coverage, similarity in matches:
            print(f"- {disease} (Coverage: {coverage:.2f}, Similarity: {similarity:.2f})")
