import numpy as np
import theano
import theano.tensor as T
import lasagne

import matplotlib.pyplot as plt

import pickle
import os
from sklearn.cross_validation import train_test_split

from prepare_images import prep_image
from cnn_creator import create_cnn
from trainer import train

X = []
y = []

for cls in range(10):
    for fn in os.listdir('./leedsbutterfly/classes/{}'.format(cls+1)):
        _, im = prep_image('./leedsbutterfly/classes/{}/{}'.format(cls+1, fn))
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

net = create_cnn()
model, loss_history, train_acc_history, val_acc_history, predict_fn = train(net, X_train, y_train, X_val, y_val, 
                                                                decay_after_epochs=2, batch_size=15, num_epochs=60)

#save the model and solver for later use
with open('saved_model2.pkl', 'wb') as output:
    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    
plt.plot(train_acc_history)
plt.plot(val_acc_history)
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

print("best validation accuracy after training for 60 epochs: ",  max(val_acc_history))
y_test_pred = predict_fn(X_test)
print(np.mean(y_test==y_test_pred))
