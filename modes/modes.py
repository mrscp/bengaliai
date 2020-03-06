from lib.config import Config
from time import time
from datetime import timedelta


class Mode(Config):
    def __init__(self):
        super(Mode, self).__init__()

        self._start_time = time()

    def __del__(self):
        print("{} finish".format(self.__class__.__name__))
        print("\tTime taken: {}".format(timedelta(seconds=(time() - self._start_time))))
