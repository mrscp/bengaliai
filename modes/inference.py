from modes.modes import Mode
from dataset.dataset import GrayImageData, CSVData
from os.path import join
from dl.cnn import CNN
import pandas as pd


class Inference(Mode):
    def __init__(self):
        super(Inference, self).__init__()
        print("Training the model")
        dataset_location = self["MAIN"]["DATASET"]

        images = GrayImageData(join(dataset_location, "test-{}".format(self["MAIN"]["IMAGE_VERSION"])))
        # print(images)
        labels = CSVData(join(dataset_location, "test.csv"), batch_size=int(self["TRAIN"]["BATCH_SIZE"]))
        inference_data, _ = labels.next_batch()
        image_names = inference_data["image_id"].apply(lambda x: x+".png").unique()

        inference_image = images.next_batch(image_names)
        inference_image = inference_image.reshape((-1, 1, inference_image.shape[1], inference_image.shape[2]))

        model = CNN()
        model.load_weights(self["MAIN"]["MODEL_LOCATION"])

        grapheme_root, vowel_diacritic, consonant_diacritic = model.predict(inference_image)
        output = {}
        for image, gr, vd, cd in zip(image_names, grapheme_root, vowel_diacritic, consonant_diacritic):
            output[image[:-4]] = {
                "grapheme_root": gr.item(),
                "vowel_diacritic": vd.item(),
                "consonant_diacritic": cd.item()
            }

        sample_submission = pd.read_csv(join(dataset_location, "sample_submission.csv"))
        targets = []
        for idx, row in sample_submission.iterrows():
            row_id_split = row["row_id"].split("_")
            image_name = "{}_{}".format(row_id_split[0], row_id_split[1])
            component_name = "{}_{}".format(row_id_split[2], row_id_split[3])
            targets.append(output[image_name][component_name])

        print(sample_submission.shape)
        sample_submission["target"] = targets
        print(sample_submission)
        print("Saving output..")
        sample_submission.to_csv(
            join(dataset_location, "output", "output_{}.csv".format(self["MAIN"]["IMAGE_VERSION"])),
            index=False
        )
