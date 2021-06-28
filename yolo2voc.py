from pascal_voc_io import XML_EXT
from pascal_voc_io import PascalVocWriter
from yolo_io import YoloReader
import os.path
import sys
import cv2

imgFolderPath = sys.argv[1]
ext = ['.jpg','.JPG','.jpeg','.JPEG']
# Search all yolo annotation (txt files) in this folder
for file in os.listdir(imgFolderPath):
    if file.endswith(tuple(ext)):
        print("Convert", file)
        image_no_extension = os.path.splitext(file)[0]
        imagePath = os.path.join(imgFolderPath, file)
        
        image=cv2.imread(imagePath)
        try:
            imageShape=list(image.shape)
        except:
            print(imagePath)
        imgFolderName = os.path.basename(imgFolderPath)
        imgFileName = os.path.basename(imagePath)

        writer = PascalVocWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)

        # Read YOLO file
        txtPath = os.path.join(imgFolderPath, image_no_extension + ".txt")
        YoloParseReader = YoloReader(txtPath, image)
        shapes = YoloParseReader.getShapes()
        num_of_box = len(shapes)

        for i in range(num_of_box):
            label = shapes[i][0]
            xmin = shapes[i][1][0][0]
            ymin = shapes[i][1][0][1]
            x_max = shapes[i][1][2][0]
            y_max = shapes[i][1][2][1]

            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

        writer.save(targetFile= imgFolderPath + "/" + image_no_extension + ".xml")
