import pandas as pd
import cv2 as cv
from os.path import join
import numpy as np


class GrayImageData:
    """
    Reads gray scaled images using open cv based on given image names in specific location

    Attributes
    ----------
    _location: str
        Location of the folder that has the images

    Methods
    -------
    next_batch(self, image_names):
        Reads and return the images
    """
    def __init__(self, location):
        self._location = location

    def next_batch(self, image_names):
        """
        Get a batch of images

        :param image_names: list
            names of the images in the given location

        :return: np.array
            Array of images
        """
        images = []
        for image_name in image_names:
            images.append(cv.imread(join(self._location, image_name), cv.IMREAD_GRAYSCALE))

        return np.array(images)


class CSVData:
    """
    Read CSV file and get data batch wise

    Attributes
    ----------
    _location: str
        Location of the CSV file
    _batch_size: int
        Number of rows will be provided in a single batch
    _data: pd.DataFrame
        Contains the data of the CSV file

    _current_location: int
        Pointer to serve rows in next batch, it is the starting position of next batch
    __max_location: int
        Number of rows in the CSV file

    Methods
    -------
    __end_location(self):
        Calculates ending location of a batch

    __current_location(self):
        Calculates the next batch starting location

    next_batch(self):
        Serves a batch of data
    """
    def __init__(self, location, batch_size):
        self._location = location
        self._batch_size = batch_size
        self._data = pd.read_csv(self._location)

        self._current_location = 0
        self.__max_location = self._data.shape[0]

    def __end_location(self):
        """
        Calculate ending location of a batch
        :return: int
            Ending index of a batch
        """
        end_location = self._current_location + self._batch_size
        return end_location if end_location <= self.__max_location else self.__max_location

    def __current_location(self):
        """
        Calculates the next batch starting location

        :return: int
            Index of next batch starting location
        """
        current_location = self.__end_location()
        if current_location < self.__max_location:
            return current_location, False
        else:
            return 0, True

    def next_batch(self):
        """
        Get a batch of data
        :return: (pd.DataFrame, boolean)
            Tuple of batch of data and is next batch location reset or not
        """
        end_location = self.__end_location()
        data = self._data.iloc[self._current_location:end_location]
        self._current_location, reset = self.__current_location()

        return data, reset
