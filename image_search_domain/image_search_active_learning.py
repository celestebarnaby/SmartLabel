from image_edit_domain.image_edit_active_learning import ImageEditActiveLearning

from image_search_domain.image_search_benchmarks import image_search_benchmarks
from image_search_domain.image_search_synthesis import ImageSearchSynthesizer
from image_search_domain.image_search_synthesis import ImageSearchInterpreter

from constants import * 

class ImageSearchActiveLearning(ImageEditActiveLearning):
    def __init__(self, semantics, question_selection):
        super().__init__(semantics, question_selection)
        self.dataset_to_program_space = {}
        self.max_prog_space_size = IMAGE_SEARCH_INIT_PROG_SPACE_SIZE

    def set_benchmarks(self):
        self.benchmarks = image_search_benchmarks 

    def set_synthesizer(self):
        self.synth = ImageSearchSynthesizer(self.semantics)

    def set_interpreter(self):
        self.interp = ImageSearchInterpreter()