"""Download the models for the demo into the data/ folder."""
from __future__ import absolute_import, division, print_function
import urllib
import hashlib
from os import makedirs
from os.path import abspath, exists, join
import sys


def report_hook(num_blocks, chunk_size, total_size, display_length=80):
    """A progress report hook as defined by urllib's urlretrieve function."""
    # Calculate progress for file
    downloaded_size = float(min(total_size, num_blocks * chunk_size))
    percentage = downloaded_size / total_size

    # Create progress line
    length = int(display_length * percentage)
    characters = '=' * length
    spacing = ' ' * (display_length - length)
    line = '[%s%s] (%0.2f%%)' % (characters, spacing, 100.0 * percentage,)

    # Write progress line with /r for in-line updates
    sys.stdout.write('\r%s' % (line, ))
    sys.stdout.flush()


def md5_hash(filepath):
    """Calculate the MD5 hash of a file, if exists, else return None."""
    if not exists(filepath):
        return None

    # Use MD5 hash to verify downloaded file
    sha = hashlib.md5()
    with open(filepath, 'rb') as f:
        for line in f:
            sha.update(line)
        return sha.hexdigest()


def ensure_models():
    # Model URLs and associated MD5 hashes
    model_url_list = [
        ('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat',
            'f72d927587ca4c97fbd165ec9cb9997f'),
        ('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
            '106118b7cf60435e6d8e04f6a6dc3657'),
    ]

    # Create the data folder that the models are saved to, if it does not exist
    if not exists('data'):
        makedirs('data')

    # Download each model and verify MD5 hash
    for model_url, model_hash in model_url_list:
        model_filename = model_url.split('/')[-1]
        model_filepath = abspath(join('data', model_filename))

        # If file does not exist or the hash is invalid, (re)download model file
        if md5_hash(model_filepath) != model_hash:
            print('Downloading %r...' % (model_filename, ))
            urllib.urlretrieve(model_url, model_filepath,
                               reporthook=report_hook)
            # print('downloaded.')

        # Verify the model file once more after downloaded
        args = (model_filepath, )
        message = 'Invalid MD5 hash of %r, please rerun this script' % args
        assert md5_hash(model_filepath) == model_hash, message
        print('Verified %r (md5 = %s)' % (model_filename, model_hash, ))


if __name__ == '__main__':
    ensure_models()
