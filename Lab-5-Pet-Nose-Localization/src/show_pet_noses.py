
import csv
import os
import argparse
import cv2
import sys
from tkinter import *
# import tkFileDialog
from tkinter.filedialog import askopenfilename

scale = 1

if __name__ == '__main__':

    # choose input image folder
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    labels_file = askopenfilename(filetypes = [("Text files","*.txt")])
    # print(labels_file)
    path = os.path.dirname(labels_file)
    # print(path)
    root.destroy()

    with open(labels_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row2 in reader:
            noseString = row2['nose']
            noseString = noseString[1:len(noseString) - 1]
            nose = tuple(map(int, noseString.split(',')))
            # imageFile = row2['image']
            noseImageFile = row2['image']
            print(noseImageFile, nose)
            imageFile = os.path.join(path, noseImageFile)
            if os.path.isfile(imageFile):
                image = cv2.imread(imageFile)
                dim = (int(image.shape[1] / scale), int(image.shape[0] / scale))
                imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                cv2.circle(imageScaled, nose, 2, (0, 0, 255), 1)
                cv2.circle(imageScaled, nose, 8, (0, 255, 0), 1)
                cv2.imshow(noseImageFile, imageScaled)
                key = cv2.waitKey(0)
                cv2.destroyWindow(noseImageFile)
                if key == ord('q'):
                    exit(0)