class Anchors():
    grid = (4, 12)
    # anchor_grid = (16,48)

    # anchor_shapes = [(100,100),(100,200),(200,100),(300,75),(75,300)]
    # anchor_shapes = [(200,200),(200,400),(400,200),(300,150),(150,300)]
    # anchor_shapes = [(100,100)]
    min_range = (100,100)
    max_range = (376,710)
    # shapes = [(50,50),(100,100),(100,200),(200,100),(200,200),(200,300),(300,200)]
    shapes = [(150,150)]

    def calc_anchor_centers(self, image_shape, anchor_grid):
        dy = int(image_shape[0]/anchor_grid[0])
        dx = int(image_shape[1]/anchor_grid[1])

        centers = []
        for y_idx in range(anchor_grid[0]):
            for x_idx in range(anchor_grid[1]):
                center_y = int((y_idx+1)*dy - dy/2)
                center_x = int((x_idx+1)*dx - dx/2)
                centers += [(center_y, center_x)]

        return centers

    def get_anchor_ROIs(self, image, anchor_centers, anchor_shapes):
        ROIs = []
        boxes = []

        for j in range(len(anchor_centers)):
            center = anchor_centers[j]

            for k in range(len(anchor_shapes)):
                anchor_shape = anchor_shapes[k]
                pt1 = [int(center[0] - (anchor_shape[0]/2)), int(center[1] - (anchor_shape[1]/2))]
                pt2 = [int(center[0] + (anchor_shape[0]/2)), int(center[1] + (anchor_shape[1]/2))]

                # pt1 = [max(0, pt1[0]), min(pt1[1], image.shape[1])]
                # pt2 = [max(0, pt2[0]), min(pt2[1], image.shape[1])]
                pt1 = [max(0, pt1[0]), max(0, pt1[1])]
                pt2 = [min(pt2[0],  image.shape[0]), min(pt2[1], image.shape[1])]

                # print('break 777: ', pt1, pt2)
                ROI = image[pt1[0]:pt2[0],pt1[1]:pt2[1],:]
                ROIs += [ROI]
                boxes += [(pt1,pt2)]

        return ROIs, boxes

    def calc_IoU(self, boxA, boxB):
        # print('break 209: ', boxA, boxB)
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0][1], boxB[0][1])
        yA = max(boxA[0][0], boxB[0][0])
        xB = min(boxA[1][1], boxB[1][1])
        yB = min(boxA[1][0], boxB[1][0])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[1][1] - boxA[0][1] + 1) * (boxA[1][0] - boxA[0][0] + 1)
        boxBArea = (boxB[1][1] - boxB[0][1] + 1) * (boxB[1][0] - boxB[0][0] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def calc_max_IoU(self, ROI, ROI_list):
        max_IoU = 0
        for i in range(len(ROI_list)):
            max_IoU = max(max_IoU, self.calc_IoU(ROI, ROI_list[i]))
        return max_IoU