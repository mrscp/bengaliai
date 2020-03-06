from modes.modes import Mode


class Process(Mode):
    def __init__(self):
        super(Process, self).__init__()
        print("process")
