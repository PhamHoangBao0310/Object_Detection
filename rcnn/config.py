import os
import glob


# Base path to original data folder
DATA_FOLDER = ".\data"
ORIGINAL_IMAGES_FOLDER = os.path.join(DATA_FOLDER, "images")
ANNOTATIONS_FOLDER = os.path.join(DATA_FOLDER, "annotations")

# Path for dataset
BASE_PATH = ".\\rcnn\dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "racoon"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_racoon"])

# Define max proposal for selective search for training and inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# DEfine config max positive and negative for each image
MAX_POSITIVE = 20
MAX_NEGATIVE = 5

# Define input dims for model CNN
INPUT_DIMS = (224, 224)
MIN_PROBA = 0.99



if __name__ == '__main__':
  print(ORIGINAL_IMAGES_FOLDER)
  print(ANNOTATIONS_FOLDER)
  print(POSITVE_PATH)
  print(NEGATIVE_PATH)
  # print(glob.glob("data\images\*.jpg"))