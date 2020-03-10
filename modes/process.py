from modes.modes import Mode
import pandas as pd
import numpy as np
from os.path import join
from pre_process.process_image import crop_resize, HEIGHT, WIDTH, save_image


class Process(Mode):
    def __init__(self):
        super(Process, self).__init__()
        print("Processing images..")
        dataset_location = self["MAIN"]["DATASET"]

        train = ['train_image_data_0.parquet',
                 'train_image_data_1.parquet',
                 'train_image_data_2.parquet',
                 'train_image_data_3.parquet']

        for parquet_name in train:
            print("processing: {}".format(parquet_name))

            df = pd.read_parquet(join(dataset_location, parquet_name), engine='pyarrow')
            n_images = df.shape[0]

            for idx in range(n_images):
                image_id = df.iloc[idx, 0]
                image = 255 - df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
                image = (image * (255.0 / image.max())).astype(np.uint8)
                image = crop_resize(image)

                save_image(join(dataset_location, "images-{}".format(self["MAIN"]["IMAGE_VERSION"]), "{}.png".format(image_id)), image)

                if idx % int(self["PROCESS"]["REPORT"]) == 0:
                    print("done processing: {} images".format(idx))
                    print("time taken: {}".format(self.elapsed_time()))
                    print()
