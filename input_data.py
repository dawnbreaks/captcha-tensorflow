# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np
import json


def load_data(data_dir, flatten=False):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    meta_info = os.path.join(data_dir, 'meta.json')
    with open(meta_info, 'r') as f:
        meta = json.load(f)

    return (
        meta,
        DataSet(
            *_read_images_and_labels(train_dir, flatten=flatten, **meta)),
        DataSet(
            *_read_images_and_labels(test_dir, flatten=flatten, **meta)),
    )


class DataSet:
    """提供 next_batch 方法"""

    def __init__(self, images, labels):
        self._images = images
        self._labels = labels

        self._num_examples = images.shape[0]

        self.ptr = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, size=100, shuffle=True):
        if self.ptr + size > self._num_examples:
            self.ptr = 0

        if self.ptr == 0:
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

        self.ptr += size
        return (
            self._images[self.ptr - size: self.ptr],
            self._labels[self.ptr - size: self.ptr],
        )


def _read_images_and_labels(dir_name, flatten, ext='.png', **meta):
    images = []
    labels = []
    for fn in os.listdir(dir_name):
        if fn.endswith(ext):
            fd = os.path.join(dir_name, fn)
            succ, image = _read_image(fd, flatten=flatten, **meta)
            if succ == False:
                continue;
            images.append(image)
            labels.append(_read_lable(fd, **meta))

    labelArray = np.array(labels)
    imageArray = np.array(images)
    return imageArray, labelArray


def _read_image(filename, flatten, width, height, **extra_meta):
    im = Image.open(filename).convert('L')
    data = np.asarray(im)
    if data.shape != (height, width):
        print 'bad image shape (%d, %d)' % data.shape
        return False, data
    if flatten:
        return data.reshape(width * height)
    return True, data


def _read_lable(filename, label_choices, num_per_image, **extra_meta):
    basename = os.path.basename(filename)
    captch_text = basename.split('_')[0]
    label_choices_len = len(label_choices)
    data = np.zeros(label_choices_len * num_per_image)
    i = 0
    for c in captch_text:
        idx = label_choices.index(c)
        data[i * label_choices_len + idx] = 1
        ++i
    return data


def display_info(meta, train_data, test_data):
    print '=' * 20
    for k, v in meta.items():
        print '%s: %s' % (k, v)
    print '=' * 20

    print 'train images: %s, labels: %s' % (train_data.images.shape, train_data.labels.shape)

    print 'test images: %s, labels: %s' % (test_data.images.shape, test_data.labels.shape)

    batch_xs, batch_ys = train_data.next_batch(100)
    print 'batch images: %s, labels: %s' % (batch_xs.shape, batch_ys.shape)


if __name__ == '__main__':
    ret1 = load_data('images/char-4-groups-1/')
    display_info(*ret1)

    # ret2 = load_data('images/char-1-groups-1000/', flatten=True)
    # display_info(*ret2)
