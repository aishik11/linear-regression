# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""A dataset loader for imports85.data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import os
import tensorflow as tf

try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
  pass




# Order is important for the csv-readers, so we use an OrderedDict here.





defaults = collections.OrderedDict([
    ("portfolio_id", [""]),
    ("desk_id", [""]),
    ("office_id", [""]),
    ("pf_category", [""]),
    ("start_date", [0.0]),
    ("sold", [0.0]),
    ("country_code", [""]),
    ("euribor_rate", [0.0]),
    ("currency", [""]),
    ("libor_rate", [0.0]),
    ("bought", [0.0]),
    ("creation_date", [0.0]),
    ("indicator_code", [""]),
    ("sell_date", [0.0]),
    ("type", [""]),
    ("hedge_value", [""]),
    ("status", [""]),
    


])# pyformat: disable

types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())


def _get_imports85():
  #path = tf.contrib.keras.utils.get_file(URL.split("/")[-1], URL)

  with open("test.csv",'r') as f:
    with open("testf.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)

  path= os.getcwd()+"\\testf.csv"
  return path


def dataset(y_name="return", train_fraction=1):
  """Load the imports85 data as a (train,test) pair of `Dataset`.

  Each dataset generates (features_dict, label) pairs.

  Args:
    y_name: The name of the column to use as the label.
    train_fraction: A float, the fraction of data to use for training. The
        remainder will be used for evaluation.
  Returns:
    A (train,test) pair of `Datasets`
  """
  # Download and cache the data
  path = _get_imports85()

  # Define how the lines of the file should be parsed
  def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    #label = features_dict.pop(y_name)

    return features_dict#, label

  def has_no_question_marks(line):
    """Returns True if the line of text has no question marks."""
    # split the line into an array of characters
    chars = tf.string_split(line[tf.newaxis], "").values
    # for each character check if it is a question mark
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question

    return no_question

  def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later. Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` beacuse `not` only works on python
    # booleans but we are dealing with symbolic tensors.
    return ~in_training_set(line)

  base_dataset = (tf.contrib.data
                  # Get the lines from the file.
                  .TextLineDataset(path)
                  # drop lines with question marks.
                  .filter(has_no_question_marks))

  train = (base_dataset
           # Take only the training-set lines.
           .filter(in_training_set)
           # Cache data so you only read the file once.
           .cache()
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line))

  # Do the same for the test-set.
  #test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train#, test
