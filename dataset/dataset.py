import pandas as pd
import cv2 as cv
from os.path import join
import numpy as np


class GrayImageData:
    def __init__(self, location):
        self._location = location

    def next_batch(self, image_names):
        images = []
        for image_name in image_names:
            images.append(cv.imread(join(self._location, image_name), cv.IMREAD_GRAYSCALE))

        return np.array(images)


class CSVData:
    def __init__(self, location, batch_size):
        self._location = location
        self._batch_size = batch_size
        self._data = pd.read_csv(self._location)

        self._current_location = 0
        self.__max_location = self._data.shape[0]

    def __end_location(self):
        end_location = self._current_location + self._batch_size
        return end_location if end_location <= self.__max_location else self.__max_location

    def __current_location(self):
        current_location = self.__end_location()
        if current_location < self.__max_location:
            return current_location, False
        else:
            return 0, True

    def next_batch(self):
        end_location = self.__end_location()
        data = self._data.iloc[self._current_location:end_location]
        self._current_location, reset = self.__current_location()

        return data, reset
