import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import glob

actual       =  np.array([])
predictions  =  np.array([])
# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
my_image = face_recognition.load_image_file("Me.jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]
known_faces = [
    my_face_encoding,
]

# Load an image with an unknown face
for img in glob.iglob("*.jpg"):
    try:
        unknown_image = face_recognition.load_image_file(img)

        # Find all the faces and face encodings in the unknown image
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_faces, unknown_face_encoding,0.55)

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_faces, unknown_face_encoding)

        if img == "Me_1.jpg" or img == "Me_2.jpg" or img == "Me_3.jpg" or img == "Me_4.jpg" or img == "Me_5.jpg" or img == "Me_6.jpg" or img == "Me_7.jpg" or img == "Me_8.jpg":
            actual = np.block([actual, 1])
            predictions = np.block([predictions, 1-face_distances])
        else:
            actual = np.block([actual, 0])
            predictions = np.block([predictions, 1-face_distances])
    except IndexError:
        print("I wasn't able to locate any faces in {}".format(img))

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)


plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'m--')
plt.xlim([0,1])
plt.ylim([0,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
