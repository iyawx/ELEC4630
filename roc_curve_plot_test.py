import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import glob

actual       =  np.array([])
predictions  =  np.array([])

# Load the jpg files into numpy arrays
my_image = face_recognition.load_image_file("Me.jpg")
for img in glob.iglob("*.jpg"):
    unknown_image = face_recognition.load_image_file(img)

# Get the face encodings for each face in each image file
# Since there could be more than one face in each image, it returns a list of encodings.
# But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
    try:
        my_face_encoding = face_recognition.face_encodings(my_image)[0]
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    except IndexError:
        print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
        quit()

    known_faces = [
        my_face_encoding,
    ]

# results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding,0.6)
    print("Is the unknown face a picture of ME? {}".format(results))
    if True in results:
        pred_results = 1
    else: 
        pred_results = 0
    print(pred_results)
    if img == "Me.jpg" or img == "unknown_test10.jpg":
        actual = np.block([actual, 1])
        predictions = np.block([predictions, pred_results])
    else:
        actual = np.block([actual, 0])
        predictions = np.block([predictions, pred_results])
print(actual)
print(predictions)

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

