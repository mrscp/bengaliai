from modes.modes import Mode
from dataset.dataset import GrayImageData, CSVData
from os.path import join
from dl.cnn import CNN
from keras.utils import to_categorical


class Train(Mode):
    def __init__(self):
        super(Train, self).__init__()
        print("Training the model")
        dataset_location = self["MAIN"]["DATASET"]

        images = GrayImageData(join(dataset_location, "images-{}".format(self["MAIN"]["IMAGE_VERSION"])))
        labels = CSVData(join(dataset_location, "train.csv"), batch_size=int(self["TRAIN"]["BATCH_SIZE"]))

        model = CNN()

        train = True

        epoch = 0
        batch = 0

        while train:
            try:
                batch_labels, reset = labels.next_batch()
                batch_images = images.next_batch(batch_labels["image_id"].apply(lambda x: x+".png").values)
                batch_images = batch_images.reshape((-1, 1, batch_images.shape[1], batch_images.shape[2]))

                grapheme_root = batch_labels["grapheme_root"].values
                grapheme_root = to_categorical(grapheme_root, 168)

                vowel_diacritic = batch_labels["vowel_diacritic"].values
                vowel_diacritic = to_categorical(vowel_diacritic, 11)

                consonant_diacritic = batch_labels["consonant_diacritic"].values
                consonant_diacritic = to_categorical(consonant_diacritic, 7)

                outputs = [grapheme_root, vowel_diacritic, consonant_diacritic]

                loss = model.train_on_batch(batch_images, outputs)

                if batch % int(self["TRAIN"]["REPORT"]) == 0:
                    print("epochs {} batches {}".format(epoch+1, batch+1))
                    print("loss:", loss)
                    model.test_on_batch(batch_images, outputs, verbose=1)
                    print("time taken: {}".format(self.elapsed_time()))
                    print()

                if reset:
                    epoch += 1

                if epoch >= int(self["TRAIN"]["EPOCHS"]):
                    train = False

                batch += 1
            except KeyboardInterrupt:
                train = False

        print("saving weights")
        model.save_weights(self["MAIN"]["MODEL_LOCATION"])
