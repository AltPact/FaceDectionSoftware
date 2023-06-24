from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.ensemble import VotingClassifier
import numpy as np
import os
import cv2

# Load images from FERET database
# image_dir = ("images/FERET/colorferet/dvd1/data/images")
# images = []
# labels = []
# for person_dir in os.listdir(image_dir):
#     person_path = os.path.join(image_dir, person_dir)
#     if os.path.isdir(person_path):
#         for filename in os.listdir(person_path):
#             if filename.endswith('.ppm'):
#                 image_path = os.path.join(person_path, filename)
#                 try:
#                     image = cv2.imread(image_path)
#                     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                     images.append(image)
#                     labels.append(person_dir)
#                 except:
#                     pass

# Load images from FERET database
image_dir = ("images/ORL")
images = []
labels = []
for person_dir in os.listdir(image_dir):
    person_path = os.path.join(image_dir, person_dir)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            image_path = os.path.join(person_path, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(person_dir)


print(f'images: {images}')
print(f'labels: {labels}')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load data and split into training and testing sets
# X_train, y_train = load_training_data()
# X_test, y_test = load_testing_data()
# X_train = X_train[0]
# X_test = X_test[0]

X_train = X_train[1]
X_test = X_test[1]

# X_train[0] = X_train[0] / 255
# X_test[0] = X_test[0] / 255


print(f'X_train: {X_train}')
print(f'X_test: {X_test}')
print(f'y_train: {y_train}')
print(f'y_test: {y_test}')
# Apply PCA to reduce dimensionality of training and testing data
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Build CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(768, 512, 1)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(num_classes, activation='softmax'))
cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train CNN model on original training data
cnn_model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Predict classes for testing data using CNN model
y_pred_cnn = cnn_model.predict_classes(X_test)

# Convert training and testing labels to categorical format
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Train PCA model on reduced training data
pca_model = PCA(n_components=50)
X_train_pca_reduced = pca_model.fit_transform(X_train_pca)
X_test_pca_reduced = pca_model.transform(X_test_pca)

# Build MLP model using reduced PCA features
mlp_model = Sequential()
mlp_model.add(Dense(128, input_shape=(50,), activation='relu'))
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(num_classes, activation='softmax'))
mlp_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train MLP model on reduced PCA features
mlp_model.fit(X_train_pca_reduced, y_train_cat, batch_size=32, epochs=10, verbose=1)

# Predict classes for testing data using MLP model
y_pred_mlp = mlp_model.predict_classes(X_test_pca_reduced)

# Create voting ensemble classifier
ensemble_model = VotingClassifier(estimators=[('cnn', cnn_model), ('mlp', mlp_model)], voting='soft')
ensemble_model.fit(X_train_pca, y_train)

# Predict classes for testing data using voting ensemble model
y_pred_ensemble = ensemble_model.predict(X_test_pca)

# Calculate accuracy of each individual model and voting ensemble model
acc_cnn = np.mean(y_pred_cnn == y_test)
acc_mlp = np.mean(y_pred_mlp == y_test)
acc_ensemble = np.mean(y_pred_ensemble == y_test)

print('CNN accuracy:', acc_cnn)
print('MLP accuracy:', acc_mlp)
print('Ensemble accuracy:', acc_ensemble)
