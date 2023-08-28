# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Input data pipeline."""
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from absl import logging
from functools import partial
import data_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE


def deconstruct_dict(batch_dict):
    key = 'inputs'
    return batch_dict[key]


def normalize_dataset(batch, data_min, data_max):
    """Normalize dataset to range [-1, 1]."""
    batch = (batch - data_min) / (data_max - data_min)
    batch = 2. * batch - 1.
    return batch


def slice_transform(batch, slice_idx=None):

    batch = tf.gather(batch, slice_idx, axis=-1)
    return batch


def data_transform(batch, problem='vae', pca=None):
  """Data transform.

  Args:
    batch: A batch of data samples.
    pca: PCA transform object.
  
  Returns:
    Transformed batch array.
  """
  if problem == 'mnist':
    batch = tf.reshape(batch, (batch.shape[0], -1))
    batch = tf.cast(batch, tf.float32) / 255.
    batch = 2. * batch - 1.

  if pca is not None:
    if batch.ndim > 2:
      init_shape = batch.shape
      batch = batch.reshape(batch.shape[0], -1)
      batch = pca.transform(batch)
      batch = batch.reshape(*init_shape)
    else:
      batch = pca.transform(batch)

  return batch


def inverse_data_transform(batch,
                           normalize=True,
                           pca=None,
                           data_min=0.,
                           data_max=1.,
                           slice_ckpt=None,
                           dim_weights=None,
                           out_channels=512):
  """Inverse data transform.

  Args:
    batch: Transformed batch array.
    pca: PCA transform object.

  Returns:
    Original batch array.
  """
  slice_idx = data_utils.load(os.path.expanduser(slice_ckpt)) if slice_ckpt else None

  if normalize:
    batch = (batch + 1.) / 2.
    batch = (data_max - data_min) * batch + data_min

  if pca is not None:
    batch = pca.inverse_transform(batch)

  if slice_idx is not None:
    transformed = np.random.randn(*batch.shape[:-1], out_channels)
    transformed[..., slice_idx] = batch
    batch = transformed

  if dim_weights is not None:
    batch = batch / dim_weights

  return batch


def get_dataset(dataset, data_shape, batch_size, normalize, slice_ckpt, include_cardinality):


    shape = tuple(map(int, data_shape))

    ds = data_utils.get_tf_record_dataset(
        file_pattern=f'{dataset}/train-*.tfrecord',
        shape=shape,
        batch_size=batch_size,
        shuffle=True,
        tokens=False)

    num_elements = ds.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
    test_percentage = 0.1
    test_size = int(test_percentage * num_elements)

    test_ds = ds.take(test_size)
    train_ds = ds.skip(test_size)

    # Batch.
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)

    train_ds = train_ds.map(partial(deconstruct_dict), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(partial(deconstruct_dict), num_parallel_calls=AUTOTUNE)


    # Slice + weight transform
    slice_idx = data_utils.load(os.path.expanduser(slice_ckpt)) if slice_ckpt else None
    train_ds = train_ds.map(partial(slice_transform, slice_idx=slice_idx), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(partial(slice_transform, slice_idx=slice_idx), num_parallel_calls=AUTOTUNE)


    # Dataset normalization.
    train_min, train_max = 0., 1.
    test_min, test_max = 0., 1.

    if normalize:
        logging.info('Normalizing dataset to have range [-1, 1].')
        config_name = "test"

        train_min, train_max = data_utils.compute_dataset_min_max(
            train_ds,
            ds_split='train',
            cache=False,
            cache_dir=os.path.expanduser(dataset),
            config=config_name)
        test_min, test_max = data_utils.compute_dataset_min_max(
            test_ds,
            ds_split='train',
            cache=False,
            cache_dir=os.path.expanduser(dataset),
            config=config_name)

        train_ds = train_ds.map(lambda example: normalize_dataset(
            example, train_min, train_max),
                                num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.map(lambda example: normalize_dataset(
            example, test_min, test_max),
                                num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.prefetch(AUTOTUNE)
        test_ds = test_ds.prefetch(AUTOTUNE)

        setattr(train_ds, 'min', train_min)
        setattr(train_ds, 'max', train_max)

        setattr(test_ds, 'min', test_min)
        setattr(test_ds, 'max', test_max)


    if include_cardinality:
        t0 = time.time()
        config_name = str(batch_size)
        data_utils.compute_dataset_cardinality(
            train_ds,
            ds_split='train',
            cache=True,
            cache_dir=os.path.expanduser(dataset),
            config=config_name)

        config_name = str(batch_size)
        data_utils.compute_dataset_cardinality(
            test_ds,
            ds_split='test',
            cache=True,
            cache_dir=os.path.expanduser(dataset),
            config=config_name)


        logging.info('Computed dataset cardinality in %f seconds', time.time() - t0)

    return train_ds, test_ds
