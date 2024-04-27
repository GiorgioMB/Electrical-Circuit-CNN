import os
import cv2
import pandas as pd

for i in range(1, 25):
    img_folder = f"archive/drafter_{i}/images/"
    csv_folder = f"archive/drafter_{i}/csvs/"

    for file in os.listdir(img_folder):
        csv = pd.read_csv(csv_folder + file.split(".")[0] + ".csv")
        image = cv2.imread(img_folder + file)
        cv2.imwrite("train/" + file.split(".")[0] + ".jpg", image)
        csv.to_csv("train/" + file.split(".")[0] + ".csv")


for file in os.listdir("archive/drafter_0/images/"):
    image = cv2.imread("archive/drafter_0/images/" + file)
    cv2.imwrite(
        "archive/drafter_00/" + file.split(".")[0] + ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )

img_folder = "archive/drafter_00/"
csv_folder = "archive/drafter_0/csvs/"

for file in os.listdir(img_folder):
    try:
        csv = pd.read_csv(csv_folder + file.split(".")[0] + ".csv")
        image = cv2.imread(img_folder + file)
        cv2.imwrite("train/" + file.split(".")[0] + ".jpg", image)
        csv.to_csv("train/" + file.split(".")[0] + ".csv")
    except:
        pass

for i in range(25, 26):
    img_folder = f"archive/drafter_{i}/images/"
    csv_folder = f"archive/drafter_{i}/csvs/"

    for file in os.listdir(img_folder):
        csv = pd.read_csv(csv_folder + file.split(".")[0] + ".csv")
        image = cv2.imread(img_folder + file)
        cv2.imwrite("train/" + file.split(".")[0] + ".jpg", image)
        csv.to_csv("train/" + file.split(".")[0] + ".csv")
