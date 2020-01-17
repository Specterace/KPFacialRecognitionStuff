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

############Deploying a model to classify face embeddings##############
#####Creating labelling for the faces passed and loaded in################
#load in face embeddings to fit the model with
data = load('kp-employees-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, text=%d' % (trainX.shape[0], testX.shape[0]))
#normalize input vectors (the face embedding vectors)
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
tesX = in_encoder.transform(testX)
#label encode targets (converting the string target variables to integers)
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

############fitting the model/linear SVM with the training data encoded#################
#fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
###evaluating the model##########
#predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
#score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
#summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

##########plotting original face and the prediction#############
#########fitting the model with the test data loaded in############
####The dataset and the embeddings must be related. The embeddings must be of the given dataset######
#load faces
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
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

#test model on a random example from the test Dataset
#if wanted, can modify code here to take a random image and evaluate it#####
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])

#take an image, extract a face, get it's embedding, store it's class/name
#if it is another image, no need to encode it (not yet)

# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
