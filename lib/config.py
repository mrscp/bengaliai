from configparser import ConfigParser
from os.path import join
import os
import sys


class Config(ConfigParser):
    """
    This class is used to represent whole configuration of the project

    Attributes
    ----------
    environment: str
        variable for which environment should be loaded
    project_dir: str
        Project path
    __config_location: str
        config file location
    """
    def __init__(self, environment="local"):
        super().__init__(comment_prefixes='/', allow_no_value=True)
        self.environment = environment
        self.project_dir = os.path.dirname(sys.modules['__main__'].__file__)
        self.__config_location = join(self.project_dir, "config.{}.ini".format(self.environment.lower()))
        self.read(self.__config_location)

    def save(self):
        with open(self.__config_location, "w") as file:
            self.write(file)
