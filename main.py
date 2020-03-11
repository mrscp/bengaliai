from modes.modes import Mode
from modes.process import Process
from modes.train import Train
from modes.inference import Inference


class Main(Mode):
    def __init__(self):
        super(Main, self).__init__()

        if self["MAIN"]["MODE"] == "process":
            Process()
        elif self["MAIN"]["MODE"] == "train":
            Train()
        elif self["MAIN"]["MODE"] == "inference":
            Inference()
        else:
            print("Invalid mode")


if __name__ == '__main__':
    Main()
