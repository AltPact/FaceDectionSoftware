import numpy as np
import cv2
import os
import tensorflow as tf

# Load images from FERET database
image_dir = ("images/FERET/colorferet/dvd1/data/images")
images = []
labels = []
i = 0
for person_dir in os.listdir(image_dir):
    person_path = os.path.join(image_dir, person_dir)
    if os.path.isdir(person_path):
        for filename in os.listdir(person_path):
            if filename.endswith('.ppm'):
                image_path = os.path.join(person_path, filename)
                try:
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    images.append(image)
                    labels.append(person_dir)
                except:
                    pass
    i=i+1
    if i % 10:
        print(i)
    if i > 200:
        break

# Convert images and labels to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f'X_train: {X_train}')
print(f'X_test: {X_test}')
print(f'Y_train: {y_train}')
print(f'Y_test: {y_test}')

# X_train = np.array(X_train, dtype=np.float32)
# X_test = np.array(X_test, dtype=np.float32)

from sklearn.preprocessing import normalize
# Normalize image data to range [0, 1]
# new_array = np.empty((6247, 768, 512), dtype=np.float32)
# X_train = X_train.astype(np.float16)
# new_array[:] = X_train

# X_train = X_train / 255
# X_test = X_test / 255

# X_train = normalize(X_train, axis=1, norm='l1')
# X_test = normalize(X_test, axis=1, norm='l1')

X_train = X_train.astype('float16') / 255.0
X_test = X_test.astype('float16') / 255.0

print(X_train)
print(X_test)

# Convert labels to one-hot encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
onehot_encoder = OneHotEncoder(sparse=False)
y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
y_test = onehot_encoder.transform(y_test.reshape(-1, 1))

# Define CNN model architecture
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import load_model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(768, 512, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.summary()

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train.reshape(-1, 768, 512, 1), y_train, epochs=10, batch_size=32, validation_data=(X_test.reshape(-1, 768, 512, 1), y_test))

# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 768, 512, 1), y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

cap = cv2.VideoCapture(1)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = model.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()