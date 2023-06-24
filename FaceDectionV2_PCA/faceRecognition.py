import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# from algoClass import mainAlgorithm
from PCA import pca_class
from TwoDPCA import twoDPcaClass
from TwoD_Square_PCA import twoDSquarePcaClass

from imageMatrix import imageToMatrixClass
from images_matrix_for_2d_square_pca import imagesToMatrixClassForTwoD
from dataset import datasetClass

import os

# Algorithm types (pca, 2d-pca, 2d2-pca)
algorithmType = "pca"

#single image = 0
#video = 1
#group image = 2
reco_type = 0

#Number of images for Training, the other images in the folder will be used for testing
no_of_images_of_one_person = 8
dataset_obj = datasetClass(no_of_images_of_one_person)

#Data for training
imgNamesTraining = dataset_obj.imgPathTraining
labelsTraining = dataset_obj.labelsTraining
NumImagesTraining = dataset_obj.NumImagesTraining
# imagesTarget = dataset_obj.imagesTargetArray
imagesTargetArray = dataset_obj.imagesTargetArray

#Data for Testing
imgPathTesting = dataset_obj.imgPathTesting
labelsTesting = dataset_obj.labelsTesting
NumImagesTesting = dataset_obj.NumImagesTesting


imgWidth, imgHeight = 50, 50
imgWidthYale, imgHeightYale = 192, 168
imageToMatrixClassObj = imageToMatrixClass(imgNamesTraining, imgWidth, imgHeight)
# imageToMatrixClassObj = imageToMatrixClass(imgNamesTraining, imgWidthYale, imgHeightYale)
imgMatrix = imageToMatrixClassObj.get_matrix()

trainingStartTime = time.process_time()

# Turning Features on and off
imgToDisplaySwitch = False

if algorithmType == "pca":
    i_t_m_c = imageToMatrixClass(imgNamesTraining, imgWidth, imgHeight)
else:
    i_t_m_c = imagesToMatrixClassForTwoD(imgNamesTraining, imgWidth, imgHeight)

scaledFace = i_t_m_c.get_matrix()

if algorithmType == "pca":
    i_t_m_c = imageToMatrixClass(imgNamesTraining, imgWidth, imgHeight)
    cv2.imshow("Original Image", cv2.resize(np.array(np.reshape(scaledFace[:,1],[imgHeight, imgWidth]), dtype = np.uint8),(200,200)))
    # cv2.waitKey()
else:
    i_t_m_c = imagesToMatrixClassForTwoD(imgNamesTraining, imgWidth, imgHeight)
    cv2.imshow("Original Image", cv2.resize((scaledFace[0]),(200,200)))
    # cv2.waitKey()

#Algorithm class
if algorithmType == "pca":
    currentAlgorithmObj = pca_class(scaledFace, labelsTraining, imagesTargetArray, NumImagesTraining, 90)
elif algorithmType == "2d-pca":
    currentAlgorithmObj = twoDPcaClass(scaledFace, labelsTraining, imagesTargetArray)
else:
    currentAlgorithmObj = twoDSquarePcaClass(scaledFace, labelsTraining, imagesTargetArray)


# currentAlgorithmObj = mainAlgorithm(imgMatrix, labelsTraining, imagesTargetArray, NumImagesTraining, imgWidth, imgHeight, qualityPercent=90)
new_coordinates = currentAlgorithmObj.reduce_dim()
# currentAlgorithmObj.show_eigen_faces(imgWidth, imgHeight, 50, 150, 0)

if algorithmType == "pca":
    currentAlgorithmObj.show_eigen_face(imgWidth, imgHeight, 50, 150, 0)

if algorithmType == "pca":
    cv2.imshow("After PCA Image", cv2.resize(np.array(np.reshape(currentAlgorithmObj.original_data(new_coordinates[1, :]), [imgHeight, imgHeight]), dtype = np.uint8), (200, 200)))
    # cv2.waitKey()
else:
    cv2.imshow("After PCA Image", cv2.resize(np.array(currentAlgorithmObj.original_data(new_coordinates[0]), dtype = np.uint8), (200, 200)))
    # cv2.waitKey()

trainingTime = time.process_time() - trainingStartTime

#Recognition Process
#Single Image
if reco_type == 0:
    time_start = time.process_time()

    # Keeping track of time and success rate of algorithms
    correctCounter = 0
    wrongCounter = 0
    i = 0
    netTimeOfReco = 0
    confidenceLevelarr = np.empty(79, dtype=float)
    confidenceLevel2arr = np.empty(79, dtype=float)
    for imgPath in imgPathTesting:
        timeStart = time.process_time()
        findedName, confidenceLevel, confidenceLevel2 = currentAlgorithmObj.recognize_face(currentAlgorithmObj.new_cord(imgPath, imgHeight, imgWidth))
        timeElapsed = (time.process_time() - timeStart)
        netTimeOfReco += timeElapsed
        rec_labels = labelsTesting[i]
        rec_name = imagesTargetArray[rec_labels]
        confidenceLevelarr[i] = confidenceLevel
        confidenceLevel2arr[i] = confidenceLevel2
        if findedName is rec_name:
            correctCounter += 1
            print("Correct ", "Name:", findedName)
        else:
            wrongCounter += 1
            print(f'Wrong: Real Name: {rec_name} Rec Label: {rec_labels} Found Name: {findedName}')
            # print("Wrong: ", "Real Name:", rec_name, "Rec Y:", rec_labels, "Found Name:", findedName)
        i+= 1

        print("i = ", i)
    

    print("Correct Counter: ", correctCounter)
    print("Wrong Counter: ", wrongCounter)
    print("Total Test Images", i)
    print("Percent of correct Identification: ", correctCounter/i*100)
    print("Total Person: ", len(imagesTargetArray))
    print("Total Train Images: ", no_of_images_of_one_person * len(imagesTargetArray))
    print("Total Time Taken for reco:", timeElapsed)
    print("Time Taken for one reco:", timeElapsed/i)
    print("Training Time: ", trainingTime)

    cv2.waitKey()
    x = list(range(1,77))
    y= confidenceLevel2arr
    plt.plot(x,y)
    plt.xlabel('Test Images')
    plt.ylabel('')
    plt.title("Relative Confidence Levels")
    plt.show()

    x = list(range(1,77))
    y= confidenceLevelarr
    plt.plot(x,y)
    plt.xlabel('Test Images')
    plt.ylabel('')
    plt.title("Absolute Confidence Levels")
    plt.show()

    # x = np.linspace(0,76,100)
    # y = confidenceLevelarr
    # w = 

#For Live Video
if reco_type == 1:
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        #Gray the image for easier face recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #For detecting the face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=7)
        i = 0
        #Draws the border around people's faces
        for(x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            scaled = cv2.resize(roi_gray, (imgHeight, imgWidth))
            rec_color = (255, 0, 0)
            rec_stroke = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), rec_color, rec_stroke)

            new_cord = currentAlgorithmObj.new_cord_for_image(scaled)
            name, confidenceLevel, confidenceLevel2 = currentAlgorithmObj.recognize_face(new_cord)
            print(name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = (255, 255, 255)
            font_stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, font_color, font_stroke, cv2.LINE_AA)

            # Adding an image overlay
            if imgToDisplaySwitch == True:
                imgToDisplay = ""
                # dir_path = os.path.join(dataset_obj.dir, '/', name)
                for img_name in os.listdir(dataset_obj.dir + '/' + name):
                    img_path = os.path.join(dataset_obj.dir + '/' + name,img_name)
                    if img_name.lower().endswith(('.jpg', '.png')):
                        imgToDisplay = img_path
                        break

                classified_face = cv2.imread(imgToDisplay)
                # Resize classified face to match size of detected face
                resized_classified_face = cv2.resize(classified_face, (w, h))

                # Add classified face overlay
                frame[y:y+h, x:x+w] = resized_classified_face


        cv2.imshow('Colored Frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#For videos
if reco_type == 4:
    # face_cascade = cv2.CascadeClassifier('cascades/data/harrcascade_frontal_alt2.xml')
    # face_cascade = cv2.CascadeClassifier("C:\\Users\\ademp\\OneDrive\\Documents\\2022 Complete Projects\\2022-Complete-Projects\\FaceDectionSoftware\\FaceDectionV2_PCA\\cascades\\data\\harrcascade_frontal_alt2.xml")
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'harrcascade_frontal_alt2.xml')  

    # #Make Black and white
    # img = cv2.imread(dir + "group.jpg", 0)

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=3)

        for (x,y,w,h) in faces:
            # Put the Squares around where the detected face is
            roi = img[y: y+h, x:x+h]
            scaled = cv2.resize(roi, (imgWidth, imgHeight))
            rec_color = (0, 255, 0)
            rec_stroke = 3
            cv2.rectangle(img, (x,y), (x+w, y+h), rec_color, rec_stroke)

            #Recognizes face
            new_cord = currentAlgorithmObj.new_cord(scaled)
            name, confidenceLevel, confidenceLevel2 = currentAlgorithmObj.recognize_face(new_cord)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_color = (255, 0, 0)
            font_stroke = 3
            cv2.putText(img, name, (x,y), font, 5, font_color, font_stroke, cv2.LINE_AA)
            # cv2.putText(img, name, (x,y), font, 1, font_color, font_stroke, cv2.LINE_AA)
        
        frame = cv2.resize(img, (1080, 568))
        cv2.imshow("Frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#For Group photos
if reco_type == 2:
    face_cascade = cv2.CascadeClassifier('cascades/data/harrcascade_frontal_alt2.xml')
    dir = "images/GroupImages/"    

    #Make Black and white
    frame = cv2.imread(dir + "group.jpg", 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    i = 0
    for (x,y,w,h) in faces:
        # Put the Squares around where the detected face is
        roi = gray[y: y+h, x:x+h]
        scaled = cv2.resize(roi, (imgWidth, imgHeight))
        rec_color = (0, 255, 0)
        rec_stroke = 3
        cv2.rectangle(frame, (x,y), (x+w, y+h), rec_color, rec_stroke)

        #Recognizes face
        new_cord = currentAlgorithmObj.new_cord(scaled)
        name, confidenceLevel, confidenceLevel2 = currentAlgorithmObj.recognize_face(new_cord)
        font = cv2.FONT_HERSHEY_COMPLEX
        font_color = (255, 0, 0)
        font_stroke = 5
        cv2.putText(frame, name + str(i), (x,y), font, 5, font_color, font_stroke, cv2.LINE_AA)
        i += 1
    
    frame = cv2.resize(frame, (1080, 568))
    cv2.imshow("Frame", frame)
    cv2.waitKey()

#For Recorded Video
if reco_type == 3:
    # face_cascade = cv2.CascadeClassifier('cascades/data/harrcascade_frontal_alt2.xml')
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    dir = "images/Videos/"
    cap=cv2.VideoCapture(dir + "testVideo.mp4", 0)

    face_locations = []
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        # Convert the image from BGR color (which OpenCV uses) to RGB   
        # color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]
        # Find all the faces in the current frame of video

        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)
        for (x,y,h,w) in faces:
            roi = gray[y: y+h, x:x+h]
            scaled = cv2.resize(roi, (imgWidth, imgHeight))
            rec_color = (0, 255, 0)
            rec_stroke = 3
            cv2.rectangle(rgb_frame, (x,y), (x+w, y+h), rec_color, rec_stroke)

            #Recognizes face
            new_cord = currentAlgorithmObj.new_cord(scaled)
            name, confidenceLevel, confidenceLevel2 = currentAlgorithmObj.recognize_face(new_cord)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_color = (255, 0, 0)
            font_stroke = 5
            cv2.putText(rgb_frame, name + str(i), (x,y), font, 5, font_color, font_stroke, cv2.LINE_AA)
            i += 1

        # Display the resulting image
        cv2.imshow('Video', frame)        

        # Wait for Enter key to stop
        if cv2.waitKey(25) == 13:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# if reco_type == "image":
#     correctCounter = 0
#     wrongCounter = 0
#     i = 0

#     for imgPath in imgPathTesting:
#         print("Testing - Image Path: ", imgPath)
#         img = currentAlgorithmObj.img_from_path(imgPath)
#         currentAlgorithmObj.show_images("Recognize Image", img)
#         # new_cord_for_image = currentAlgorithmObj.new_cord(imgPath, imgWidth, imgHeight)

#         findedName = currentAlgorithmObj.recognize_face(currentAlgorithmObj.new_cord(imgPath, imgWidth, imgHeight))
#         targetIndex = labelsTesting[i]
#         print("Target Index: ", targetIndex)
#         # originalName = imagesTargetArray[targetIndex]
#         originalName = imagesTargetArray[i]
#         if findedName is originalName:
#             correctCounter += 1
#             print("Correct Result", " Name: ", findedName, "Original Name: ", originalName)
#         else:
#             wrongCounter += 1
#             # print("Wrong Result", "Found Name: ", findedName, "Original Name: ", originalName)
#         i += 1
#         print("i = ", i)
    
#     print("Total Correct", correctCounter)
#     print("Total Wrong", wrongCounter)
#     print("Percentage", correctCounter/(correctCounter + wrongCounter) * 100)