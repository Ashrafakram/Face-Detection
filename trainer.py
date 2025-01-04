# import os
# import cv2
# import numpy as np
# from PIL import Image
#
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# path = "dataset"
#
#
# def get_images_with_id(path):
#     images_paths=[os.path.join(path,f) for f in os.listdir(path)]
#     faces = []
#     ids = []
#     for single_image_path in images_paths:
#         faceImg = Image.open(single_image_path).convert('L')
#         faceNp = np.array(faceImg, np.uint8)
#         id = int(os.path.split(single_image_path)[-1].split(".")[1])
#         print(id)
#         faces.append(faceNp)
#         ids.append(id)
#         cv2.imshow("Trainingmodel", faceNp)
#         cv2.waitKey(10)
#     return np.array(ids), faces
#
# ids, faces = get_images_with_id(path)
# recognizer.train(faces, id)
# recognizer.save("recognizer/trainingdatas.yml")
#
# cv2.destroyAllWindows()


import os
import cv2
import numpy as np
from PIL import Image

# Create the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"

def get_images_with_id(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faces = []
    ids = []
    for single_image_path in images_path:
        faceImg = Image.open(single_image_path).convert('L')  # Convert to grayscale
        faceNp = np.array(faceImg, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        print(f"Processing ID: {id}")
        faces.append(faceNp)
        ids.append(id)
        cv2.imshow("Training Model", faceNp)
        cv2.waitKey(10)
    return np.array(ids), faces

# Ensure the 'recognizer' directory exists
if not os.path.exists("recognizer"):
    os.makedirs("recognizer")

# Train the recognizer
ids, faces = get_images_with_id(path)
ids = np.array(ids)  # Convert to NumPy array
recognizer.train(faces, ids)
recognizer.save("recognizer/trainingdatas.yml")

cv2.destroyAllWindows()
