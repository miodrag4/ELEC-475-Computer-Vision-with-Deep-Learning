
import json
import csv
import os
import argparse
import cv2
from tkinter import Tk, filedialog

keyPt = None
image = None
image_filename = None
clone = None
ptSelected = None
scale = None

def click_and_pick(event, x, y, flags, param):
    # grab references to the global variables
    global keyPt, image, image_filename, clone, ptSelected, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        ptSelected = True

        if flags == cv2.EVENT_LBUTTONDOWN:
            keyPt = [x, y]
        elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_ALTKEY:
            keyPt[0] += 1
        elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY:
            keyPt[0] -= 1
        elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_ALTKEY + cv2.EVENT_FLAG_SHIFTKEY:
            keyPt[1] += 1
        elif flags == cv2.EVENT_LBUTTONDOWN + cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_SHIFTKEY:
            keyPt[1] -= 1

        dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.circle(imageScaled, tuple(keyPt), int(2), (int(0),int(0),int(255)), int(1))
        cv2.circle(imageScaled, tuple(keyPt), int(10), (int(0),int(255),int(0)), int(1))
        cv2.imshow(image_filename, imageScaled)

    elif event == cv2.EVENT_RBUTTONDOWN:
        ptSelected = False
        image = clone.copy()
        dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(image_filename, imageScaled)

def main():

    global keyPt, image, image_filename, clone, ptSelected, scale

    theta = 0.0
    # keyPt = None
    fileList = []
    noseList = []
    scale = 1.0
    outFile = None

    # choose input image folder
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    inImagePath = filedialog.askdirectory(title='Select input image directory')
    print(inImagePath)
    root.destroy()
    outFile = os.path.join(inImagePath, 'labels.txt')

    for filename in os.listdir(inImagePath):
        filepath = os.path.join(inImagePath, filename)
        if os.path.isfile(filepath) and (filename.endswith('.png') or filename.endswith('.bmp') or filename.endswith('.jpg')):
            fileList = fileList + [filename]

    print(fileList)

    ptList = [[]] * len(fileList)

    i = 0
    ptSelected = False

    print('press \'m\' for menu ...')
    while i < len(fileList):

        filename = fileList[i]
        filepath = inImagePath + "/" + filename

        if os.path.isfile(filepath) and (filepath.endswith('.png') or filepath.endswith('.bmp') or filepath.endswith('.jpg')):

            image = cv2.imread(filepath)

            clone = image.copy()

            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.destroyAllWindows()
            image_filename = filename
            cv2.imshow(image_filename, imageScaled)
            cv2.setMouseCallback(image_filename, click_and_pick)

            ptSelected = False

            key = cv2.waitKey(0) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("m"):
                print('--------- Menu ----------')
                print('\t\t<space> = save keypoint and load next image)')
                print('\t\t<left mouse click> = select keypoint')
                print('\t\t<right mouse click> = unselect keypoint')
                print('\t\tCTRL + <left mouse click> = move keypoint left')
                print('\t\tALT + <left mouse click> = move keypoint right')
                print('\t\tCTRL + SHIFT + <left mouse click> = move keypoint up')
                print('\t\tALT + SHIFT + <left mouse click> = move keypoint down')
                print('\t\tq = (q)uit')

            elif key == ord(">"): # next image
                i += 1

            elif key == ord("<"): # previous image
                i -= 1

            elif key == 32 : # space bar

                if ptSelected:
                    noseList = noseList + [[filename, (int(keyPt[0]/scale), int(keyPt[1]/scale))]]

                i += 1

    print('saving : ', noseList)
    filename = outFile

    with open(outFile, 'w', newline='') as csvfile:
        fieldnames = ['image', 'nose']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
        writer.writeheader()
        for idx in range(len(noseList)):
            writer.writerow(
                {'image': noseList[idx][0], 'nose': noseList[idx][1]})

# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    main()