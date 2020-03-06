# dataset_location = "D:\\datasets"
#
# print(dataset_location)
#
# with ZipFile("/media/strange/Storage/datasets/bengaliai-cv19.zip") as z:
#     with z.open("test.csv") as f:
#         train = pd.read_csv(f, header=0, delimiter="\t")
#         print(train.head())  # print the first 5 rows


class Dataset:
    def __init__(self):
        print("dataset")
