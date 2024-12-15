from abc import ABC, abstractmethod
import numpy as np

class BoardDetector(ABC):
    @abstractmethod
    def detect(self, image : np.ndarray) -> np.ndarray:
        '''
        Take an image and return an image of just the board
        '''
        pass