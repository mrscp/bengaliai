import pandas as pd
from zipfile import ZipFile


class Main:
    def __init__(self):
        dataset_location = "D:\\datasets"

        print(dataset_location)

        with ZipFile("bengaliai-cv19.zip") as z:
            with z.open("test.csv") as f:
                train = pd.read_csv(f, header=0, delimiter="\t")
                print(train.head())  # print the first 5 rows


if __name__ == '__main__':
    Main()
