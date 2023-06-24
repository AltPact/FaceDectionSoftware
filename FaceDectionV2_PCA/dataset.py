import os
from PIL import Image
import numpy as np

class datasetClass:
    def __init__(self, required_no):
        #Dataset Path
        self.dir = ("images/ORL")
        self.yaleDir = ("images/YaleDataSet/")
        self.FERETDir = ("images/FERET/colorferet/data/images")

        # Images used for training (first 8 images)
        self.imgPathTraining = []
        self.labelsTraining = []
        self.NumImagesTraining = []

        # Images used for Testing (last 2 images)
        self.imgPathTesting = []
        self.labelsTesting = []
        self.NumImagesTesting = []

        # Contains the names/labels for the classification 
        self.imagesTargetArray = []
        self.imagesTargetSet = {}

        # Choose what data set you want to use
        self.dataset = "Not"
        per_no = 0
        if (self.dataset == "FERET"):
            for name in os.listdir(self.FERETDir):
                dir_path = os.path.join(self.FERETDir, name)
                if os.path.isdir(dir_path):
                    # print("Length: ",len(os.listdir(dir_path)))
                    #Checks the folder has at least 8 images in it
                    if len(os.listdir(dir_path)) >= required_no:
                        i = 0
                        for img_name in os.listdir(dir_path):
                            img_path = os.path.join(dir_path,img_name)

                            if i < required_no:
                                self.imgPathTraining += [img_path]
                                self.labelsTraining += [per_no]
                                # Check to see if this person img is already in the self so no duplicants
                                if len(self.NumImagesTraining) > per_no:
                                    # add 1 more to training
                                    self.NumImagesTraining[per_no] += 1
                                else:
                                    self.NumImagesTraining += [1]
                                
                                if i is 0:
                                    self.imagesTargetArray += [name]
                                    self.imagesTargetSet[per_no] = name
                                
                            else:
                                self.imgPathTesting += [img_path]
                                self.labelsTesting += [per_no]

                                if len(self.NumImagesTesting) > per_no:
                                    self.NumImagesTesting[per_no] += 1
                                else:
                                    self.NumImagesTesting += [1]
                            i += 1
                        per_no += 1
            print("Per_no :", per_no)
            print("i: ", i)
            print("Label Testing array: ", self.labelsTesting)
            print("Image Target array: ", self.imagesTargetArray)
        elif (self.dataset == "Yale"):
            for name in os.listdir(self.yaleDir):
                dir_path = os.path.join(self.yaleDir, name)
                if os.path.isdir(dir_path):
                    # print("Length: ",len(os.listdir(dir_path)))
                    #Checks the folder has at least 8 images in it
                    if len(os.listdir(dir_path)) >= required_no:
                        i = 0
                        for img_name in os.listdir(dir_path):
                            img_path = os.path.join(dir_path,img_name)

                            if i < required_no:
                                self.imgPathTraining += [img_path]
                                self.labelsTraining += [per_no]
                                # Check to see if this person img is already in the self so no duplicants
                                if len(self.NumImagesTraining) > per_no:
                                    # add 1 more to training
                                    self.NumImagesTraining[per_no] += 1
                                else:
                                    self.NumImagesTraining += [1]
                                
                                if i is 0:
                                    self.imagesTargetArray += [name]
                                    self.imagesTargetSet[per_no] = name
                                
                            else:
                                self.imgPathTesting += [img_path]
                                self.labelsTesting += [per_no]

                                if len(self.NumImagesTesting) > per_no:
                                    self.NumImagesTesting[per_no] += 1
                                else:
                                    self.NumImagesTesting += [1]
                            i += 1
                        per_no += 1
            print("Per_no :", per_no)
            print("i: ", i)
            print("Label Testing array: ", self.labelsTesting)
            print("Image Target array: ", self.imagesTargetArray)
        else:
            #Looping through all the traing data that is within the "images/" folder
            for name in os.listdir(self.dir):
                dir_path = os.path.join(self.dir, name)
                if os.path.isdir(dir_path):
                    # print("Length: ",len(os.listdir(dir_path)))
                    #Checks the folder has at least 8 images in it
                    if len(os.listdir(dir_path)) >= required_no:
                        i = 0
                        for img_name in os.listdir(dir_path):
                            img_path = os.path.join(dir_path,img_name)

                            if i < required_no:
                                self.imgPathTraining += [img_path]
                                self.labelsTraining += [per_no]
                                # Check to see if this person img is already in the self so no duplicants
                                if len(self.NumImagesTraining) > per_no:
                                    # add 1 more to training
                                    self.NumImagesTraining[per_no] += 1
                                else:
                                    self.NumImagesTraining += [1]
                                
                                if i is 0:
                                    self.imagesTargetArray += [name]
                                    self.imagesTargetSet[per_no] = name
                                
                            else:
                                self.imgPathTesting += [img_path]
                                self.labelsTesting += [per_no]

                                if len(self.NumImagesTesting) > per_no:
                                    self.NumImagesTesting[per_no] += 1
                                else:
                                    self.NumImagesTesting += [1]
                            i += 1
                        per_no += 1
            print("Per_no :", per_no)
            print("i: ", i)
            print("Label Testing array: ", self.labelsTesting)
            print("Image Target array: ", self.imagesTargetArray)