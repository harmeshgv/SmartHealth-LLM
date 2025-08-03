import os
import sys
from typing import List, Tuple
from collections import defaultdict
from statistics import mean
from langchain_core.documents import Document

# Ensure project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from backend.utils.data_preprocessing import DataPreprocessing
from backend.utils.matcher import DiseaseMatcher


class Pipeline:   
    def __init__(self):
        self.preprocessor = DataPreprocessing()
        self.matcher =  DiseaseMatcher()

    def Stem(self, query : str):
        tokens = self.preprocessor.preprocess(query)

        return self.matcher.match(tokens)

     





