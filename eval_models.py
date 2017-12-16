import numpy as np
import matplotlib.pyplot as plt
import os.path

def load_model_eval(experiment):
    return np.load('{}/model_eval_{}.npz'.format(experiment, experiment))

def load_model_evals(experiment):
    return [ load_model_eval(experiment) ] + [ load_model_eval('{}-{}'.format(experiment, i)) for i in xrange(2, 5) ]

def main():

    baselines = load_model_evals('baseline')
    batch_avg_mi_ascs = load_model_evals('batch_avg_mi_asc')
    batch_avg_mi_descs = load_model_evals('batch_avg_mi_desc')
    batch_avg_mi_asc_seqs = load_model_evals('batch_avg_mi_asc_seq')
    batch_avg_mi_desc_seqs = load_model_evals('batch_avg_mi_desc_seq')

    plt.title('Training Accuracy')
    print 'Training Accuracy'
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for experiment in baselines:
        x = [acc for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [acc for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [acc for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [acc for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [acc for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('training_accuracy.eps')
    plt.close()

    plt.title('Training Loss')
    print 'Training Loss'
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for experiment in baselines:
        x = [loss for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [loss for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [loss for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [loss for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [loss for (loss, acc) in experiment['train']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('training_loss.eps')
    plt.close()

    plt.title('Test Accuracy')
    print 'Test Accuracy'
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for experiment in baselines:
        x = [acc for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [acc for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [acc for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [acc for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [acc for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('test_accuracy.eps')
    plt.close()

    plt.title('Test Loss')
    print 'Test Loss'
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    for experiment in baselines:
        x = [loss for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [loss for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [loss for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [loss for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [loss for (loss, acc) in experiment['test']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('test_loss.eps')
    plt.close()

    plt.title('Bootstrap Mean Accuracy')
    print 'Bootstrap Mean Accuracy'
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    for experiment in baselines:
        x = [acc for (loss, acc) in experiment['bootstrap_mean']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [acc for (loss, acc) in experiment['bootstrap_mean']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [acc for (loss, acc) in experiment['bootstrap_mean']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [acc for (loss, acc) in experiment['bootstrap_mean']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [acc for (loss, acc) in experiment['bootstrap_mean']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('bootstrap_mean.eps')
    plt.close()

    plt.title('Bootstrap Variance')
    print 'Bootstrap Variance'
    plt.xlabel('Epochs')
    plt.ylabel('Variance')
    for experiment in baselines:
        x = [acc*acc for (loss, acc) in experiment['bootstrap_std']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [acc*acc for (loss, acc) in experiment['bootstrap_std']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [acc*acc for (loss, acc) in experiment['bootstrap_std']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [acc*acc for (loss, acc) in experiment['bootstrap_std']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [acc*acc for (loss, acc) in experiment['bootstrap_std']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('bootstrap_variance.eps')
    plt.close()

    plt.title('First Layer to Output MI')
    print 'First Layer to Output MI'
    plt.xlabel('Epochs')
    plt.ylabel('MI')
    for experiment in baselines:
        x = [layers[0] for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [layers[0] for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [layers[0] for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [layers[0] for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [layers[0] for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('first_layer_to_output.eps')
    plt.close()

    plt.title('Average Layer to Output MI')
    print 'Average Layer to Output MI'
    plt.xlabel('Epochs')
    plt.ylabel('MI')
    for experiment in baselines:
        x = [np.mean(layers) for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [np.mean(layers) for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [np.mean(layers) for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [np.mean(layers) for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [np.mean(layers) for layers in experiment['y_mi']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('average_layer_to_output.eps')
    plt.close()

    plt.title('Average Sequential Layer MI')
    print 'Average Sequential Layer MI'
    plt.xlabel('Epochs')
    plt.ylabel('MI')
    for experiment in baselines:
        x = [np.mean(layers) for layers in experiment['seq_mi']]
        plt.plot(range(len(x)), x, 'b')
        print 'Baseline: {}'.format(x)
    plt.plot(range(len(x)), x, 'b', label='Baseline')
    for experiment in batch_avg_mi_ascs:
        x = [np.mean(layers) for layers in experiment['seq_mi']]
        plt.plot(range(len(x)), x, 'g')
        print 'Y MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'g', label='Y MI Ascending')
    for experiment in batch_avg_mi_descs:
        x = [np.mean(layers) for layers in experiment['seq_mi']]
        plt.plot(range(len(x)), x, 'c')
        print 'Y MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'c', label='Y MI Descending')
    for experiment in batch_avg_mi_asc_seqs:
        x = [np.mean(layers) for layers in experiment['seq_mi']]
        plt.plot(range(len(x)), x, 'r')
        print 'Seq. MI Asc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'r', label='Seq. MI Ascending')
    for experiment in batch_avg_mi_desc_seqs:
        x = [np.mean(layers) for layers in experiment['seq_mi']]
        plt.plot(range(len(x)), x, 'm')
        print 'Seq. MI Desc.: {}'.format(x)
    plt.plot(range(len(x)), x, 'm', label='Seq. MI Descending')
    plt.legend(loc=0)
    plt.savefig('average_sequent_layer.eps')
    plt.close()

if __name__ == '__main__':
    main()
