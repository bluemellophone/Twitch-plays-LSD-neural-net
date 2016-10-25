"""Convert the vlfeat matlab .mat model weight files into Numpy .npy files."""
from __future__ import absolute_import, division, print_function
from download_models import ensure_models
import numpy as np
import scipy.io


def process_model(model_tag, idxs):
    data = scipy.io.loadmat('data/imagenet-vgg-verydeep-%s.mat' % (model_tag, ))

    print('\nProcessing: %r' % (model_tag, ))
    print(data.keys())

    print('Params:')
    params = []
    for i in idxs:
        W = data['layers'][0][i][0][0][2][0][0]
        W = np.transpose(W, (3, 2, 0, 1))
        b = data['layers'][0][i][0][0][2][0][1].reshape(-1)
        # Invert kernels
        # W = W[:, :, ::-1, ::-1]
        print('\t', W.shape, b.shape)
        params.extend([W, b])

    mean = data['meta'][0][0][2][0][0][2]
    classes = data['meta'][0][0][1][0][0][1][0]

    print('Mean:')
    print('\t', mean.shape)

    print('Classes:')
    print('\t', classes.shape)

    np.save('data/vgg%s.npy' % (model_tag, ), params)
    np.save('data/mean-%s.npy' % (model_tag, ), mean)
    np.save('data/classes-%s.npy' % (model_tag, ), classes)
    if model_tag in [16]:
        np.save('data/mean.npy', mean)
        np.save('data/classes.npy', classes)

if __name__ == '__main__':
    # Ensure we have the models locally
    ensure_models()

    model_dict = {
        16: [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 33, 35],
        19: [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34, 37, 39, 41],  # NOQA
    }

    for model_tag in sorted(model_dict.keys()):
        idxs = model_dict[model_tag]
        process_model(model_tag, idxs)
