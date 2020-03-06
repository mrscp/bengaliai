from modes.modes import Mode
from modes.process import Process


class Main(Mode):
    def __init__(self):
        super(Main, self).__init__()

        if self["MAIN"]["MODE"] == "process":
            Process()


if __name__ == '__main__':
    Main()
