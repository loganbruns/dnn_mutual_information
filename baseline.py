import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np
from keras import backend
from keras.utils import np_utils
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, GaussianDropout
from keras.datasets import mnist
from sklearn.metrics import mutual_info_score
import time

def create_model(dset):
    model = Sequential()
    model.add(Dense(units=256, input_dim=dset.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(units=128))
    model.add(Activation('relu'))
    model.add(Dense(units=96))
    model.add(Activation('relu'))
    model.add(Dense(units=64))
    model.add(Activation('relu'))
    model.add(Dense(units=32))
    model.add(Activation('relu'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
    return model

def compute_y_mi_for_layer(layers_output, layer, bins=128):
    y = np.digitize(layers_output[len(layers_output)-1], np.linspace(0, 1, bins))
    l = np.digitize(layers_output[layer], np.linspace(np.min(layers_output[layer]), np.max(layers_output[layer]), bins))
    y_quantizied, y_idx = np.unique(y, axis=0, return_inverse=True)
    layer_quantizied, layer_idx = np.unique(l, axis=0, return_inverse=True)
    return mutual_info_score(None,None,contingency=np.histogram2d(layer_idx, y_idx)[0])

def compute_seq_mi_across_layers(layers_output, layer, bins=128):
    l = np.digitize(layers_output[layer], np.linspace(np.min(layers_output[layer]), np.max(layers_output[layer]), bins))
    n = np.digitize(layers_output[layer+1], np.linspace(np.min(layers_output[layer+1]), np.max(layers_output[layer+1]), bins))
    layer_quantizied, layer_idx = np.unique(l, axis=0, return_inverse=True)
    next_quantizied, next_idx = np.unique(n, axis=0, return_inverse=True)
    return mutual_info_score(None,None,contingency=np.histogram2d(layer_idx, next_idx)[0])

def sort_batches(x_train, y_train, offset, batch_size, dump_layers, reverse=False):
    batch_mi_list = range(offset / batch_size)
    for batch in range(offset / batch_size, x_train.shape[0] / batch_size):
        batch_offset = batch * batch_size
        x_train_subset = x_train[batch_offset:batch_offset+batch_size]
        y_train_subset = y_train[batch_offset:batch_offset+batch_size]
        layers_output = dump_layers([x_train_subset, 0])
        batch_mi = np.mean([compute_seq_mi_across_layers(layers_output, 2*i-1) for i in range(1, len(layers_output)/2)])
        batch_mi_list += [batch_mi]
    batches = np.argsort(batch_mi_list)    
    if reverse:
        batches = batches[::-1]
    p = [batch*batch_size + i for batch in batches for i in xrange(batch_size)]
    x_train = x_train[p, :]
    y_train = y_train[p, :]
    return x_train, y_train

def main():
    np.random.seed(100)
    experiment = 'baseline'

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float').reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])/255.)
    y_train = np_utils.to_categorical(y_train, 10)
    x_test = (x_test.astype('float').reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])/255.)
    y_test = np_utils.to_categorical(y_test, 10)

    model = create_model(x_train)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    dump_layers = backend.function([model.input]+ [backend.learning_phase()], [layer.output for layer in model.layers])
    print model.summary()
    with open('{}/model_{}.json'.format(experiment, experiment), 'w') as json_file:
        json_file.write(model.to_json())    

    batch_size = 32
    y_mi_list = []
    acc_list = []
    loss_list = []
    for epoch in range(100):
        start = time.time()
        for batch in xrange(x_train.shape[0] / batch_size):
            offset = batch * batch_size
            # x_train, y_train = sort_batches(x_train, y_train, offset, batch_size, dump_layers)
            x_train_subset = x_train[offset:offset+batch_size]
            y_train_subset = y_train[offset:offset+batch_size]
            history = model.fit(x_train_subset, y_train_subset, batch_size=batch_size, epochs=1)
            acc_list += history.history['acc']
            loss_list += history.history['loss']
            layers_output = dump_layers([x_train, 0])
            y_mi = [compute_y_mi_for_layer(layers_output, 2*i-1) for i in range(1, len(layers_output)/2)]
            y_mi_list += [y_mi]
            print "{} MI for batch {}".format(y_mi, batch)
            if (batch % 100) == 0:
                model.save_weights('{}/model_{}_{}_{}.h5'.format(experiment, experiment, epoch, batch))
        finish = time.time()
        print 'Finished epoch {} with {} data points in {} seconds'.format(epoch, len(y_mi_list), finish-start)
        model.save_weights('{}/model_{}_{}.h5'.format(experiment, experiment, epoch))
        np.savez('{}/history_{}_{}.npz'.format(experiment, experiment, epoch), y_mi=y_mi_list, accuracy=acc_list, loss=loss_list)
        p = np.random.permutation(x_train.shape[0])
        x_train = x_train[p, :]
        y_train = y_train[p, :]

if __name__ == '__main__':
    main()
