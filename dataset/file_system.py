from zipfile import ZipFile


class FileSystemZip:
    def __init__(self):
        with ZipFile("/home/strange/Data/bengaliai-cv19.zip") as z:
            for f in z:
                print(f)
