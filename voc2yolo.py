from pascal_voc_io import XML_EXT
from pascal_voc_io import PascalVocWriter
from pascal_voc_io import PascalVocReader
from yolo_io import YoloReader
from yolo_io import YOLOWriter
import os.path
import sys

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage


imgFolderPath = sys.argv[1]
ext = ['.jpg','.JPG','.jpeg','.JPEG']

# Search all pascal annotation (xml files) in this folder
for file in os.listdir(imgFolderPath):
    if file.endswith(tuple(ext)):
        print("Convert", file)
        image_no_extension = os.path.splitext(file)[0]
        imagePath = os.path.join(imgFolderPath, file)

        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(), 1 if image.isGrayscale() else 3]
        imgFolderName = os.path.basename(imgFolderPath)
        imgFileName = os.path.basename(imagePath)

        writer = YOLOWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)

        # Read classes.txt
        classListPath = imgFolderPath + "/" + "classes.txt"
        classesFile = open(classListPath, 'r')
        classes = classesFile.read().strip('\n').split('\n')
        classesFile.close()

        # Read VOC file
        filePath = os.path.join(imgFolderPath, image_no_extension + ".txt")
        tVocParseReader = PascalVocReader(filePath)
        shapes = tVocParseReader.getShapes()
        num_of_box = len(shapes)

        for i in range(num_of_box):
            label = classes.index(shapes[i][0])
            xmin = shapes[i][1][0][0]
            ymin = shapes[i][1][0][1]
            x_max = shapes[i][1][2][0]
            y_max = shapes[i][1][2][1]

            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

        writer.save(targetFile= imgFolderPath + "/" + image_no_extension + ".txt")