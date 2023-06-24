import numpy as np
import cv2
import scipy.linalg as s_linalg

# Main difference between PCA and 2DPCA is that:
# 2DPCA is based on 2D matrices as opposed to the standard PCA which is based on see the images as 1D Vectors 
class twoDSquarePcaClass:

    # 
    def give_p(self, d):
        print("D", d)
        sum = np.sum(d)
        sum_85 = 0.95 * sum
        temp = 0
        p = 0
        while temp < sum_85:
            temp += d[p]
            p += 1
        return p

    # Reduce the dimensions of image
    def reduce_dim(self):
        NumOfImages = self.images.shape[0]
        MatHeight = self.images.shape[1]
        MatWidth = self.images.shape[2]
        # Generate the 2*1D arraies that will be used to evalute the feature data
        g_t = np.zeros((MatHeight, MatHeight))
        h_t = np.zeros((MatWidth, MatWidth))

        for i in range(NumOfImages):
            # temp = np.dot(self.images_mean_subtracted[i].T, self.images_mean_subtracted[i])
            g_t += np.dot(self.images_mean_subtracted[i].T, self.images_mean_subtracted[i])
            h_t += np.dot(self.images_mean_subtracted[i], self.images_mean_subtracted[i].T)

        g_t /= NumOfImages
        h_t /= NumOfImages

        #For G_T
        d_mat, p_mat = np.linalg.eig(g_t)
        p_1 = self.give_p(d_mat)
        self.new_bases_gt = p_mat[:, 0:p_1]

        #For H_T
        d_mat, p_mat = np.linalg.eig(h_t)
        p_2 = self.give_p(d_mat)
        self.new_bases_ht = p_mat[:, 0:p_2]

        new_coordinates_temp = np.dot(self.images, self.new_bases_gt)

        self.new_coordinates = np.zeros((NumOfImages, p_2, p_1))

        for i in range(NumOfImages):
            self.new_coordinates[i, :, :] = np.dot(self.new_bases_ht.T, new_coordinates_temp[i])

        return self.new_coordinates

    # Shows all the variables that are inherit to this model
    def __init__(self, images, y, targetNames):
        self.images = np.asarray(images)
        # y is a array of all the
        self.y = y
        self.targetNames = targetNames
        self.mean_face = np.mean(self.images, 0)
        self.images_mean_subtracted = self.images - self.mean_face


    def original_data(self, new_coordinates):
        return np.dot(self.new_bases_ht, np.dot(new_coordinates, self.new_bases_gt.T))

    # 
    def new_cord(self, name, imgHeight, imgWidth):
        img = cv2.imread(name, 0)
        cv2.imshow("Recognize Image",img)
        # cv2.waitKey()
        gray = cv2.resize(img, (imgHeight, imgWidth))
        return np.dot(self.new_bases_ht.T, np.dot(gray, self.new_bases_gt))

    # Check later
    # 
    def new_cord_for_image(self, image):
        return np.dot(self.new_bases_ht.T, np.dot(image, self.new_bases_gt))

    # Main Functions that calls everything else
    def recognize_face(self, new_cord):
        NumOfImages = len(self.y)
        distances = []
        sum = 0
        for i in range(NumOfImages):
            temp_imgs = self.new_coordinates[i]
            dist = np.linalg.norm(new_cord - temp_imgs)
            distances += [dist]
            sum += dist
        
        min = np.argmin(distances)
        per = self.y[min]
        per_name = self.targetNames[per]
        confidence = 100 - (distances[min] / sum)
        if distances[min] < 14975:
            print("Person", per, ":", min, self.targetNames[per], "Dist:", distances[min])
            print(f'Confidence: {confidence}%')
            return per_name, 0.0, 0.0
        else:
            print("Person", per, ":", min, 'Unknown', "Dist:", distances[min])
            return 'Un'






