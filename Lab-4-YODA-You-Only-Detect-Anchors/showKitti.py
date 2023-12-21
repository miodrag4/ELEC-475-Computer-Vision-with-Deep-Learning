import torch
import argparse
import cv2
from KittiDataset import KittiDataset

display = True

def main():

    print('running showKitti ...')
    # freeze_support()

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-m', metavar='mode', type=str, help='[train/test]')

    args = argParser.parse_args()

    input_dir = None
    if args.i != None:
        input_dir = args.i

    training = True
    if args.m == 'test':
        training = False


    min_dx = 10000
    max_dx = -1
    min_dy = 10000
    max_dy = -1

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(dir=input_dir, training=training)

    ROI_shapes = []
    ROI_dx = []
    ROI_dy = []
    i = 0
    for item in enumerate(dataset):

        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        print(i, idx, label)
        i += 1

        for j in range(len(label)):
            name = label[j][0]
            name_class = label[j][1]
            minx = int(label[j][2])
            miny = int(label[j][3])
            maxx = int(label[j][4])
            maxy = int(label[j][5])
            cv2.rectangle(image, (minx,miny), (maxx, maxy), (0,0,255))

            if name_class == 2:
                dx = maxx - minx + 1
                if dx > max_dx:
                    max_dx = dx
                if dx < min_dx:
                    min_dx = dx

                dy = maxy - miny + 1
                if dy > max_dy:
                    max_dy = dy
                if dy < min_dy:
                    min_dy = dy

                ROI_shapes += [(dx,dy)]
                ROI_dx += [dx]
                ROI_dy += [dy]

        if display == True:
            cv2.imshow('image', image)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

###################################################################

main()
