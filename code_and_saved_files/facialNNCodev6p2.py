#facial recognition code
####This only covers fitting a Linear SVM on the face embeddings for an image dataset#####
####This assumes the embeddings and the faces to be fitted/analyzed are pre-saved in ".npz" files###

#various imports
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot

from os.path import isdir
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

import sys

#needed functions (for face extraction and face encoding)
#function to extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    ##########MAYBE CHANGE THE IMAGE SIZE???#########
    #load image from file
    image = Image.open(filename)
    #convert to RGB, if needed
    image = image.convert('RGB')
    #convert to array
    pixels = asarray(image)
    #create the detector, using default weights
    detector = MTCNN()
    #detect faces in the image
    results = detector.detect_faces(pixels)
    #extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    #bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    #extract face
    face = pixels[y1:y2, x1:x2]
    #resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

#get the face embedding for one faces
def get_embedding(model, face_pixels):
    #scale pixel values
    face_pixels = face_pixels.astype('float32')
    #standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    #transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    #make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

#########fitting the model with the previously computed training data loaded in############
####The dataset and the embeddings must be related. The embeddings must be of the given dataset######
#load faces dataset
data = load('kp-employees-dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('kp-employees-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# loading in and fitting the model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

#Loading in the keras facenet model that will extract the faces from the input images
faceModel = load_model('facenet_keras.h5')
print('Face Model Loaded')

#take an image, extract a face, get it's embedding, store it's class/name
#Declare the image file name and the name/class of the image in the call
inputImg = sys.argv[1]
input_face_pixels = extract_face(inputImg)
input_face_emb = get_embedding(faceModel, input_face_pixels)
input_face_name = sys.argv[2]

#use the models to make a prediction of the face
samples = expand_dims(input_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get the name associated with the prediction
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
#print out the result and the percentage the model predicts it's answer
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
#print out the expected/actual answer
print('Expected: %s' % input_face_name)

#plot the face (optional)
#pyplot.imshow(input_face_pixels)
#title = '%s (%.3f)' % (predict_names[0], class_probability)
#pyplot.title(title)
#pyplot.show()
