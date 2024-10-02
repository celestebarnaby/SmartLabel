from abc import ABC, abstractmethod
from constants import *

class Synthesizer(ABC):
    def __init__(self, semantics):
        self.semantics = semantics 


    @abstractmethod 
    def synthesize(self, examples):
        """
        Synthesize the hypothesize space of programs that are consistent with the initial set of I/O examples.

        Returns: a list of programs in the given DSL
        """
        pass 

    def synthesize_for_learnsy(self):
        return self.synthesize([])









