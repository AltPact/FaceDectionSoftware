import sys
import os
import bz2
from bz2 import decompress
for dirpath, _, filenames in os.walk('/Users/ademp/OneDrive/Documents/2022 Complete Projects/2022-Complete-Projects/FaceDectionSoftware/FaceDectionV2_PCA/images/FERET/colorferet/dvd1/data/images/Extracting_files'):
    for filename in filenames:
        if filename.endswith('.bz2'):
            filepath = os.path.join(dirpath, filename)
            newfilepath = os.path.join(dirpath, filename[:-4])
            print(filepath)
            print(newfilepath)
            with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
                for data in iter(lambda : file.read(100 * 1024), b''):
                    new_file.write(data)


            # # Changes directory to directory with a bz2 file in it
            # # tempDir = dirpath.replace('\\\\', '\\')
            # print("Current cwd: ", os.getcwd())
            # os.chdir(str(dirpath))
            # # Extract the files using 7zip
            # os.system('7z e ' + os.path.join(dirpath, filename))