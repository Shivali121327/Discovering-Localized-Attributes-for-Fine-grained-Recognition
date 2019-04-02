import numpy as np
import theano
import theano.tensor as T
import lasagne

import matplotlib.pyplot as plt

import pickle
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

from prepare_images import prep_image
from vgg16 import build_model
from trainer import train

pre_trained_model = pickle.load(open('vgg16.pkl'))
net = build_model()
lasagne.layers.set_all_param_values(net['prob'], pre_trained_model['param values'])
IMAGE_MEAN = pre_trained_model['mean value'][:, np.newaxis, np.newaxis]

X = []
y = []

for cls in range(10):
    for fn in os.listdir('./leedsbutterfly/classes/{}'.format(cls+1)):
        _, im = prep_image('./leedsbutterfly/classes/{}/{}'.format(cls+1, fn), IMAGE_MEAN=IMAGE_MEAN)
        X.append(im)
        y.append(cls)
        
X = np.concatenate(X)
y = np.array(y).astype('int32')
print(X.shape)
print(y.shape)
# Split data into training, testing and validation set
rng = np.random.RandomState(0)
permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
train_idxs, test_idxs = train_test_split(range(len(y)),  random_state=0)
train_idxs, val_idxs = train_test_split(range(len(train_idxs)),  random_state=0)

X_train = X[train_idxs]
y_train = y[train_idxs]

X_val = X[val_idxs]
y_val = y[val_idxs]

X_test = X[test_idxs]
y_test = y[test_idxs]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
# check training data contains examples from all classes in good number
print(sum(y_train==0))
print(sum(y_train==1))
print(sum(y_train==2))
print(sum(y_train==3))
print(sum(y_train==4))
print(sum(y_train==5))
print(sum(y_train==6))
print(sum(y_train==7))
print(sum(y_train==8))
print(sum(y_train==9))

# connect output classifier to the last fully connected layer of the network
net['scores'] = lasagne.layers.DenseLayer(net['fc7_dropout'], num_units=10, 
                                          nonlinearity=lasagne.nonlinearities.softmax)
model, loss_history, train_acc_history, val_acc_history, predict_fn = train(net, X_train, y_train, X_val, y_val,
                                                                            batch_size=10, num_epochs=20)

#save the model and solver for later use
with open('saved_model1.pkl', 'wb') as output:
    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

plt.subplot(2, 1, 1)
plt.plot(np.array(loss_history).clip(max=3))
plt.xlabel('iteration')
plt.ylabel('loss')
plt.subplot(2, 1, 2)
plt.plot(train_acc_history)
plt.plot(val_acc_history)
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

print("best validation accuracy after training for 20 epochs: ",  max(val_acc_history))
species = {0:'Danaus plexippus', 1:'Heliconius charitonius', 2:'Heliconius erato', 3:'Junonia coenia', 
          4:'Lycaena phlaeas', 5:'Nymphalis antiopa', 6:'Papilio cresphontes', 7:'Pieris rapae', 
          8:'Vanessa atalanta', 9:'Vanessa cardui'}
def classify(img_name, predict_fn, species):
    X_unseen = []
    rawim, im = prep_image(img_name)
    plt.imshow(rawim)
    X_unseen.append(im)
    X_unseen = np.concatenate(X_unseen)
    y_pred = int(''.join(map(str, predict_fn(X_unseen))))
    return y_pred

y_pred = classify('unseen_danus_plexipus.jpg', predict_fn, species)
print("its %s" %(species[y_pred]))

y_pred = classify('unseenvanessacardui.jpg', predict_fn, species)
print("its %s" %(species[y_pred]))

y_test_pred = predict_fn(X_test)
print(np.mean(y_test==y_test_pred))

print(confusion_matrix(y_test, y_test_pred))
