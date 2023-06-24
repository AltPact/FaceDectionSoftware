import numpy as np
import cv2
import scipy.linalg as s_linalg


class twoDPcaClass:

    #function used for finding value of p for covering 95% of image information
    def give_p(self, d):

        #sum of all eigen values
        sum = np.sum(d)
        sum_95 = 0.95 * sum
        temp = 0
        p = 0
        while temp < sum_95:
            temp += d[p]
            p += 1
        return p

    def reduce_dim(self):

        numOfImages = self.images.shape[0]
        MatHeight = self.images.shape[1]

        #creating emptymatrix for find covarience matrix
        g_t = np.zeros((MatHeight, MatHeight))

        for i in range(numOfImages):

            #multiplying net subtracted image with its transpose and adding in gt
            temp = np.dot(self.images_mean_subtracted[i].T, self.images_mean_subtracted[i])
            g_t += temp

        #dividing by total number of images
        g_t /= numOfImages

        #finding eigen values and eigen vectors
        d_mat, p_mat = np.linalg.eig(g_t)

        #finding first p important vectors
        p = self.give_p(d_mat)
        self.new_bases = p_mat[:, 0:p]

        #finding new coordinates using dot product new bases
        self.new_coordinates = np.dot(self.images, self.new_bases)

        #returning new coordinates matrix
        return self.new_coordinates


    def __init__(self, images, y, targetNames):
        self.images = np.asarray(images)
        self.y = y
        self.targetNames = targetNames

        #finding means of image
        self.mean_face = np.mean(self.images, 0)

        #subtracting mean face from images
        self.images_mean_subtracted = self.images - self.mean_face


    def original_data(self, new_coordinates):
        return (np.dot(new_coordinates, self.new_bases.T))


    def new_cord(self, name, imgHeight, imgWidth):
        img = cv2.imread(name, 0)
        cv2.imshow("Recognize Image", img)
        # cv2.waitKey()
        gray = cv2.resize(img, (imgHeight, imgWidth))
        return np.dot(gray, self.new_bases)

    def new_cord_for_image(self, image):
        return np.dot(image, self.new_bases)


    def recognize_face(self, new_cord):
        numOfImages = len(self.y)
        distances = []
        sum = 0
        for i in range(numOfImages):
            temp_imgs = self.new_coordinates[i]
            dist = np.linalg.norm(new_cord - temp_imgs)
            distances += [dist]
            sum += dist

        # print("Distances", distances)
        min = np.argmin(distances)
        confidence = 100 - (distances[min]/sum)
        per = self.y[min]
        per_name = self.targetNames[per]

        print(f'Person {per} : {min} {self.targetNames[per]} Dist: {distances[min]}')
        return per_name, 0.0, 0.0






