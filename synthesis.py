from abc import ABC, abstractmethod
from constants import *

class Synthesizer(ABC):
    def __init__(self, semantics):
        self.semantics = semantics 


    @abstractmethod 
    def synthesize(self, examples):
        pass 









