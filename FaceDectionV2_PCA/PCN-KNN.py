import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Define function to load image dataset
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
#         if img is not None:
#             images.append(img)
#     return images

# Load images from FERET database
def load_images_from_folder(folder):
    # image_dir = ("images/ORL")
    trainImages = []
    testImages = []
    trainLabels = []
    testLabels = []
    for person_dir in os.listdir(folder):
        person_path = os.path.join(folder, person_dir)
        if os.path.isdir(person_path):
            i = 0
            for filename in os.listdir(person_path):
                image_path = os.path.join(person_path, filename)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if i < 8:
                    trainImages.append(image)
                    trainLabels.append(person_dir)
                else:
                    testImages.append(image)
                    testLabels.append(person_dir)
                i=i+1
    return trainImages, testImages, trainLabels, testLabels

# Load training data and resize images to a fixed size
image_dir = ("images/ORL")
trainImages, testImages, trainLabels, testLabels = load_images_from_folder(image_dir)
fixed_size = (100, 100)
train_images_resized = [cv2.resize(img, fixed_size) for img in trainImages]

# Convert training images to numpy array and flatten them
X_train = np.array(train_images_resized).reshape(len(train_images_resized), -1)

# Perform PCA on training data
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)

# Load testing data and resize images to a fixed size
# test_folder = 'test_images_folder'
# test_images = load_images_from_folder(test_folder)
print(trainImages)
print(testImages)
test_images_resized = [cv2.resize(img, fixed_size) for img in testImages]

# Convert testing images to numpy array and flatten them
X_test = np.array(test_images_resized).reshape(len(test_images_resized), -1)

# Perform PCA on testing data
X_test_pca = pca.transform(X_test)

# Load labels for training data
# y_train = np.loadtxt('train_labels.txt')


# Train KNN classifier on PCA-transformed training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, trainLabels)

# Predict labels for testing data using KNN classifier and PCA-transformed testing data
y_test_pred = knn.predict(X_test_pca)

# Print predicted labels for testing data
print(y_test_pred)