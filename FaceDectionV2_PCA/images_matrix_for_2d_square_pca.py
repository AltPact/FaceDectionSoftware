import cv2
import numpy as np


class imagesToMatrixClassForTwoD:

    def __init__(self, imgPath, imgWidth, imgHeight):
        self.imgPath = imgPath
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
        self.imgSize = (imgWidth * imgHeight)



    def get_matrix(self):

        img_mat = np.zeros(
            (len(self.imgPath), self.imgHeight ,self.imgWidth),
            dtype=np.uint8)

        i = 0
        for name in self.imgPath:
            # print("Name", name)
            gray = cv2.imread(name, 0)
            gray = cv2.resize(gray, (self.imgHeight, self.imgWidth))
            mat = np.asmatrix(gray)
            img_mat[i, :, :] = mat
            i += 1
        print("Matrix Size:", img_mat.shape)
        return img_mat
