import numpy as np
import cv2
import scipy.linalg as s_linalg
# import scipy.stats.norm as norm
import scipy.stats as stats 


class pca_class:

    # Keep looping through the array until the collect feature detected crosses the barrier of acceptance
    def give_p(self, d):
        sum = np.sum(d)
        sum_85 = self.QualityPercent * sum/100
        temp = 0
        p = 0
        while temp < sum_85:
            temp += d[p]
            p += 1
        return p

    def reduce_dim(self):
        p, d, q = s_linalg.svd(self.images, full_matrices=True)
        p_matrix = np.matrix(p)
        d_diag = np.diag(d)
        q_matrix = np.matrix(q)
        p = self.give_p(d)
        self.new_bases = p_matrix[:, 0:p]
        self.new_coordinates = np.dot(self.new_bases.T, self.images)
        return self.new_coordinates.T

    # Shows all the variables that are inherit to this model
    def __init__(self, images, y, targetNames, NumOfElements, QualityPercent):
        self.NumOfElements = NumOfElements
        self.images = np.asarray(images)
        self.y = y
        self.targetNames = targetNames
        mean = np.mean(self.images, 1)
        self.mean_face = np.asmatrix(mean).T
        self.images = self.images - self.mean_face
        self.QualityPercent = QualityPercent

    # Shows the orginal image
    def original_data(self, new_coordinates):
        return self.mean_face + (np.dot(self.new_bases, new_coordinates.T))

    # Shows the images as their raw eigen values (the feature data)
    def show_eigen_face(self, height, width, min_pix_int, max_pix_int, eig_no):
        ev = self.new_bases[:, eig_no:eig_no + 1]
        min_orig = np.min(ev)
        max_orig = np.max(ev)
        ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * ev)
        ev_re = np.reshape(ev, (height, width))
        cv2.imshow("Eigen Face " + str(eig_no),  cv2.resize(np.array(ev_re, dtype = np.uint8),(200, 200)))
        # cv2.waitKey()

    # Creates a new mean based on the new inputed images mean
    def new_cord(self, name, img_height, img_width):
        img = cv2.imread(name)
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img_height, img_width))
        # Ravel will take the Matrix and unravel it into a single vector to represent the image (saves computing time)
        img_vec = np.asmatrix(gray).ravel()
        # T is the transpose of the given array 
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec)/(len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    # def find_example_image(self, image):


    # 
    def new_cord_for_image(self, image):
        img_vec = np.asmatrix(image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.y)) + img_vec) / (len(self.y) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    # Calculate relative confidence score
    def relative_confidence(confidence_level, threshold):
        if confidence_level >= threshold:
            return (confidence_level - threshold) / (1 - threshold)
        else:
            return 0

    #Binomial Proportion confidence interval
    def recognize_face(self, new_cord_pca, k=0):
        classes = len(self.NumOfElements)
        start = 0
        confidence = 0
        sum = 0
        distances = []
        for i in range(classes):
            temp_imgs = self.new_coordinates[:, int(start): int(start + self.NumOfElements[i])]
            # Take away the average face features by taking away the mean
            mean_temp = np.mean(temp_imgs, 1)
            start = start + self.NumOfElements[i]
            # Below gives the Frobenius norm found here: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
            # The nuclear norm is the sum of the singular values 
            # Gives the Euclidean distances of all values (distance between 2 points in Euclidean space)
            dist = np.linalg.norm(new_cord_pca - mean_temp)
            # z_scores = (stats.zscore(new_cord_pca, axis = 1) - mean_temp)/np.std(new_cord_pca)
            distances += [dist]
            sum += dist
        z_scores = stats.zscore(distances)
        # z_score = (stats.zscore(distances) - np.mean(distances))/np.std(distances)
        # print(f'z_scores: {z_scores}')
        min = np.argmin(distances)
        # confidence_level = 1 - norm.cdf(z_scores)
        confidence = 100 - (distances[min] / sum)
        #Temp Threshold
        threshold = 100000
        confidence2 = distances[min]/threshold
        if distances[min] < threshold:
            # print("Person", k, ":", min, self.targetNames[min])
            print(f'Person {k}: {min} {self.targetNames[min]} Dist: {distances[min]}')
            print(f'Confidence: {confidence}%')
            return self.targetNames[min], confidence, confidence2
        else:
            # print("Person", k, ":", min, 'Unknown')
            print(f'Person {k}: {min} Unknown')
            return 'Unknown', confidence, confidence2