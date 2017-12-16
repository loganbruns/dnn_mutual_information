import os
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import numpy as np
from keras import backend
from keras.utils import np_utils
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, GaussianDropout
from keras.datasets import mnist
from sklearn.metrics import mutual_info_score
import os.path
import sys
import time

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
    return mutual_info_score(None, None, contingency=np.histogram2d(layer_idx, next_idx)[0])

def main():
    if len(sys.argv) != 2:
        print 'Usage: python eval_model.py <experiment>'
        sys.exit(-1)

    np.random.seed(100)
    experiment = sys.argv[1]
    experiment_dir = experiment
    batch_size = 32

    with open('{}/model_{}.json'.format(experiment_dir, experiment), 'r') as json_file:
        model_json = json_file.read()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float').reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])/255.)
    y_train = np_utils.to_categorical(y_train, 10)
    x_test = (x_test.astype('float').reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])/255.)
    y_test = np_utils.to_categorical(y_test, 10)

    epoch = 0
    train_metrics = []
    test_metrics = []
    bootstrap_mean = []
    bootstrap_std = []
    y_mi_list = []
    seq_mi_list = []
    while os.path.exists('{}/model_{}_{}.h5'.format(experiment_dir, experiment, epoch)) and epoch < 5:
        model = model_from_json(model_json)
        model.load_weights('{}/model_{}_{}.h5'.format(experiment_dir, experiment, epoch))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        dump_layers = backend.function([model.input]+ [backend.learning_phase()], [layer.output for layer in model.layers])
        train_metrics += [model.evaluate(x_train, y_train, 32)]
        test_metrics += [model.evaluate(x_test, y_test, 32)]
        metrics = []
        for run in xrange(5):
            bootstrap = np.random.choice(xrange(x_test.shape[0]), x_test.shape[0])
            metrics += [model.evaluate(x_test[bootstrap, :], y_test[bootstrap, :], 32)]
        bootstrap_mean += [np.mean(metrics, axis=0)]
        bootstrap_std += [np.std(metrics, axis=0)]
        layers_output = dump_layers([x_train, 0])
        y_mi_list += [[compute_y_mi_for_layer(layers_output, 2*i-1) for i in range(1, len(layers_output)/2)]]
        seq_mi_list += [[compute_seq_mi_across_layers(layers_output, 2*i-1) for i in range(1, len(layers_output)/2)]]
        epoch += 1

    np.savez('{}/model_eval_{}.npz'.format(experiment_dir, experiment),
             train=train_metrics, test=test_metrics,
             bootstrap_mean=bootstrap_mean, bootstrap_std=bootstrap_std,
             y_mi=y_mi_list, seq_mi=seq_mi_list)

    print '\nExperiment: {}\n'.format(experiment)
    print 'Train metrics:\n{}\n'.format(train_metrics)
    print 'Test metrics:\n{}\n'.format(test_metrics)
    print 'Bootstrap metrics means:\n{}\n'.format(bootstrap_mean)
    print 'Bootstrap metrics std:\n{}\n'.format(bootstrap_std)
    print 'Y MI:\n{}\n'.format(y_mi_list)
    print 'Seq MI:\n{}\n'.format(seq_mi_list)

if __name__ == '__main__':
    main()
    
    
