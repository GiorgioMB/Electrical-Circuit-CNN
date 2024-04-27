import os
import cv2

for i in range(1, 25):
    img_folder = f"./archive/drafter_{i}/images/"
    csv_folder = f"./archive/drafter_{i}/csvs/"

    for file in os.listdir(img_folder):
        os.replace(img_folder + file, "./train/" + file)
        os.replace(
            csv_folder + file.split(".")[0] + ".csv",
            "./train/" + file.split(".")[0] + ".csv",
        )


for file in os.listdir("./archive/drafter_0/images/"):
    image = cv2.imread("./archive/drafter_0/images/" + file)
    cv2.imwrite(
        "./archive/drafter_00/" + file.split(".")[0] + ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100],
    )

img_folder = "./archive/drafter_00/"
csv_folder = "./archive/drafter_0/csvs/"

for file in os.listdir(csv_folder):
    os.replace(
        img_folder + file.split(".")[0] + ".jpg",
        "./train/" + file.split(".")[0] + ".jpg",
    )
    os.replace(
        csv_folder + file.split(".")[0] + ".csv",
        "./train/" + file.split(".")[0] + ".csv",
    )
for i in range(25, 26):
    img_folder = f"./archive/drafter_{i}/images/"
    csv_folder = f"./archive/drafter_{i}/csvs/"

    for file in os.listdir(img_folder):
        os.replace(img_folder + file, "./test/" + file)
        os.replace(
            csv_folder + file.split(".")[0] + ".csv",
            "./test/" + file.split(".")[0] + ".csv",
        )
