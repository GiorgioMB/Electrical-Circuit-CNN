import cv2 as cv
import os

dataset_path  = "C:\\Users\\Asus\\Documents\\VS_Code_files\\Python\\CNN_Electrical_Circuit\\dataset" #Insert path to your dataset

try:
    os.mkdir(dataset_path + f"\\test") #Creating a test directory where the preprocessed images from sample directory will be stored
except FileExistsError:
    pass


# Preprocessing images using OpenCV
def image_preprocess(path: str) -> None:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE) #Transfering images to Black and White
    block_size = 11
    th = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY, block_size , 5)
    img_new = cv.imwrite(dataset_path + f"\\test\\" + os.path.basename(path), th)


#iterate through directories
def main():
    
    count = 0

    for dirpath, dirnames, filenames in os.walk(dataset_path):
        
        #Choose one directory as a sample to assess the quality
        if f"\\drafter_14\\images" in dirpath: 
            for filename in filenames:
                fpath = dirpath + '\\' + filename
                image_preprocess(fpath)
                count += 1
                print(f'Number of proccessed images: {count}')
            print("Preprocessing Complete!")

if __name__ == "__main__":
    main()










