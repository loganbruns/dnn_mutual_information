import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys

def main():
    if len(sys.argv) != 2:
        print 'Usage: python eval_history.py <experiment>'
        sys.exit(-1)

    experiment = sys.argv[1]
    experiment_dir = experiment

    epoch = 0
    while os.path.exists('{}/history_{}_{}.npz'.format(experiment_dir, experiment, epoch)):
        epoch += 1
    history = np.load('{}/history_{}_{}.npz'.format(experiment_dir, experiment, epoch-1))

    plt.title('Training Accuracy and Loss ({})'.format(experiment))
    plt.xlabel('Batches')
    plt.ylabel('Accuracy / Loss')
    plt.plot(range(len(history['loss'])), history['loss'], 'g', label='Training Loss')
    plt.plot(range(len(history['accuracy'])), history['accuracy'], 'b', label='Training Accuracy')
    plt.legend(loc=1)
    plt.savefig('{}/{}_training_accuracy_loss.eps'.format(experiment_dir, experiment))
    plt.close()

    plt.title('Mutual Information By Layer ({})'.format(experiment))
    plt.xlabel('Batches')
    plt.ylabel('Mutual Information')
    colors = ['b', 'g', 'r', 'c', 'm']
    for layer in xrange(5):
        plt.plot(range(len(history['y_mi'][:,layer])), history['y_mi'][:,layer], colors[layer], label='Mutual Information for layer {}'.format(layer))
    plt.legend(loc=0)
    plt.savefig('{}/{}_mutual_information.eps'.format(experiment_dir, experiment))
    plt.close()

if __name__ == '__main__':
    main()
