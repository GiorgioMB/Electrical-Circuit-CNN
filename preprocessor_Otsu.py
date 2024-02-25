import cv2 as cv
import os

dataset_path  = "C:\\Users\\Asus\\Documents\\VS_Code_files\\Python\\CNN_Electrical_Circuit\\dataset"

try:
    os.mkdir(dataset_path + f"\\test")
except FileExistsError:
    pass


# Preprovessing images using OpenCV
def image_preprocess(path: str) -> None:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE) #Transfering images to Black and White
    img_contrasted = cv.convertScaleAbs(img, alpha=2.5)
    img_blured = cv.GaussianBlur(img_contrasted, (5,5), 0)
    th = cv.threshold(img_blured, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
    img_new = cv.imwrite(dataset_path + f"\\test\\" + os.path.basename(path), th)


#iterate through directories
def main():
    
    count = 0

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if f"\\drafter_14\\images" in dirpath:
            for filename in filenames:
                fpath = dirpath + '\\' + filename
                image_preprocess(fpath)
                count += 1
                print(f'Number of proccessed images: {count}')
            print("Preprocessing Complete!")

if __name__ == "__main__":
    main()










