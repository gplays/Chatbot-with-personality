# Copyright 2017 Google Inc. All Rights Reserved.
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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
        collections.namedtuple("BatchedInput",
                               ("initializer", "source", "target_input",
                                "target_output", "source_sequence_length",
                                "target_sequence_length", "source_speaker",
                                "target_speaker"))):
    pass


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       src_spkr_dataset,
                       tgt_spkr_dataset,
                       spkr_table,
                       batch_size,
                       eos,
                       src_max_len=None):
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

    if src_max_len:
        src_dataset = src_dataset.map(lambda src: src[:src_max_len])
    # Convert the word strings to ids
    src_dataset = src_dataset.map(
            lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
    # Convert the spkr strings to ids
    src_spkr_dataset = src_spkr_dataset.map(
            lambda src_spkr: tf.cast(spkr_table.lookup(src_spkr),
                                     tf.int32))
    tgt_spkr_dataset = tgt_spkr_dataset.map(
            lambda tgt_spkr: tf.cast(spkr_table.lookup(tgt_spkr),
                                     tf.int32))

    # Add spkrs
    src_dataset = tf.data.Dataset.zip(
            (src_dataset, src_spkr_dataset, tgt_spkr_dataset))
    # Add in the word counts.
    src_dataset = src_dataset.map(
            lambda src, src_spkr, tgt_spkr: (
                src, tf.size(src), src_spkr, tgt_spkr))

    def batching_func(x):
        return x.padded_batch(
                batch_size,
                # The entry is the source line rows;
                # this has unknown-length vectors.  The last entry is
                # the source row size; this is a scalar.
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # src_spkr
                    tf.TensorShape([])),  # tgt_spkr
                # Pad the source sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true
                # sequence.
                padding_values=(
                    src_eos_id,  # src
                    0,
                    0,
                    0))  # src_len -- unused

    batched_dataset = batching_func(src_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, src_seq_len, src_spkr_ids,
     tgt_spkr_ids) = batched_iter.get_next()
    return BatchedInput(
            initializer=batched_iter.initializer,
            source=src_ids,
            target_input=None,
            target_output=None,
            source_sequence_length=src_seq_len,
            target_sequence_length=None,
            source_speaker=src_spkr_ids,
            target_speaker=tgt_spkr_ids)


def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 src_spkr_dataset,
                 tgt_spkr_dataset,
                 spkr_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
    tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
    tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset,
                                           src_spkr_dataset,
                                           tgt_spkr_dataset))

    src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
    if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

    src_tgt_dataset = src_tgt_dataset.shuffle(
            output_buffer_size, random_seed, reshuffle_each_iteration)

    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt, src_spkr, tgt_spkr: (
                tf.string_split([src]).values, tf.string_split([tgt]).values,
                src_spkr, tgt_spkr),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Filter zero length input sequences.
    src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt, src_spkr, tgt_spkr: tf.logical_and(
                    tf.size(src) > 0, tf.size(tgt) >
                    0))

    if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt, src_spkr, tgt_spkr: (src[:src_max_len], tgt,
                                                      src_spkr, tgt_spkr),
                num_parallel_calls=num_parallel_calls).prefetch(
                output_buffer_size)
    if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
                lambda src, tgt, src_spkr, tgt_spkr: (src, tgt[:tgt_max_len],
                                                      src_spkr, tgt_spkr),
                num_parallel_calls=num_parallel_calls).prefetch(
                output_buffer_size)
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt, src_spkr, tgt_spkr: (
                tf.cast(src_vocab_table.lookup(src), tf.int32),
                tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
                tf.cast(spkr_table.lookup(src_spkr), tf.int32),
                tf.cast(spkr_table.lookup(tgt_spkr), tf.int32)),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with
    # <eos>.
    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt, src_spkr, tgt_spkr: (
                src,
                tf.concat(([tgt_sos_id], tgt),0),
                tf.concat((tgt, [tgt_eos_id]),0),
                src_spkr, tgt_spkr),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    # Add in sequence lengths.
    src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in, tgt_out, src_spkr, tgt_spkr: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in),
                src_spkr, tgt_spkr),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
                batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.  The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # tgt_input
                    tf.TensorShape([None]),  # tgt_output
                    tf.TensorShape([]),  # src_len
                    tf.TensorShape([]),  # tgt_len
                    tf.TensorShape([]),  # src_spkr
                    tf.TensorShape([])),  # tgt_spkr
                # Pad the source and target sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true
                # sequence.
                padding_values=(
                    src_eos_id,  # src
                    tgt_eos_id,  # tgt_input
                    tgt_eos_id,  # tgt_output
                    0,  # src_len -- unused
                    0,  # tgt_len -- unused
                    0,  # src_spkr
                    0))  # tgt_spkr

    if num_buckets > 1:

        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len,u_4,u_5):
            # Calculate bucket_width by maximum source sequence length.
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs
            # with length
            # over ((num_bucket-1) * bucket_width) words all go into the last
            #  bucket.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence
            # and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width,
                                   tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(
                        key_func=key_func, reduce_func=reduce_func,
                        window_size=batch_size))

    else:
        batched_dataset = batching_func(src_tgt_dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
     tgt_seq_len, src_spkr_ids, tgt_spkr_ids) = (batched_iter.get_next())
    return BatchedInput(
            initializer=batched_iter.initializer,
            source=src_ids,
            target_input=tgt_input_ids,
            target_output=tgt_output_ids,
            source_sequence_length=src_seq_len,
            target_sequence_length=tgt_seq_len,
            source_speaker=src_spkr_ids,
            target_speaker=tgt_spkr_ids)
