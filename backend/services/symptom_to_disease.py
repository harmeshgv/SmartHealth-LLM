import os
import sys
from typing import List, Tuple
from collections import defaultdict
from statistics import mean
from langchain_core.documents import Document

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.utils.DIseaseMatcherAgent import DiseaseMatcherAgent


class Pipeline:   
    def __init__(self):
        self.matcher = DiseaseMatcherAgent()


    def Stem(self, query : str):
        matches = self.matcher.match(query, top_k=3)


        return matches

     





