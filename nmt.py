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


"""TensorFlow NMT model implementation."""
from __future__ import print_function


import argparse
import os
import random
import sys

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from . import inference
from . import train
from .utils import evaluation_utils
from .utils import misc_utils as utils
from .utils import vocab_utils

utils.check_tensorflow_version()

FLAGS = None


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_units",
                        type=int, default=32, help="Network size.")
    parser.add_argument("--num_units_speaker",
                        type=int, default=32, help="Speaker embedding size.")
    parser.add_argument("--num_layers",
                        type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers",
                        type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers",
                        type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")
    parser.add_argument("--encoder_type",
                        type=str, default="uni",
                        help="""
      uni | bi | gnmt.
      For bi, we build num_encoder_layers/2 bi-directional layers.
      For gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1)
        uni-directional layers.
        
      """)
    parser.add_argument("--residual",
                        type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to add residual connections.")
    parser.add_argument("--time_major",
                        type="bool", nargs="?", const=True,
                        default=True,
                        help="Whether to use time-major mode for dynamic RNN.")
    parser.add_argument("--num_embeddings_partitions",
                        type=int, default=0,
                        help="Number of partitions for embedding vars.")

    # attention mechanisms
    parser.add_argument("--attention",
                        type=str, default="", help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
      attention\
      """)
    parser.add_argument(
        "--attention_architecture",
        type=str,
        default="standard",
        help="""\
      standard | gnmt | gnmt_v2.
      standard: use top layer to compute attention.
      gnmt: GNMT style of computing attention, use previous bottom layer to
          compute attention.
      gnmt_v2: similar to gnmt, but use current bottom layer to compute
          attention.\
      """)
    parser.add_argument(
        "--output_attention",
        type="bool", nargs="?", const=True,
        default=True,
        help="""\
      Only used in standard attention_architecture. Whether use attention as
      the cell output at each timestep.
      .\
      """)
    parser.add_argument(
        "--pass_hidden_state",
        type="bool", nargs="?", const=True,
        default=True,
        help="""\
      Whether to pass encoder's hidden state to decoder when using an attention
      based model.\
      """)

    # optimizer
    parser.add_argument("--optimizer",
                        type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate",
                        type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--warmup_steps",
                        type=int, default=0,
                        help="How many steps we inverse-decay learning.")
    parser.add_argument("--warmup_scheme",
                        type=str, default="t2t", help="""\
      How to warmup learning rates. Options include:
        t2t: Tensor2Tensor's way, start with lr 100 times smaller, then
             exponentiate until the specified lr.\
      """)
    parser.add_argument("--decay_scheme",
                        type=str, default="",
                        help="""\
      How we decay learning rate. Options include:
        luong234: after 2/3 num train steps, we start halving the learning rate
          for 4 times before finishing.
        luong5: after 1/2 num train steps, we start halving the learning rate
          for 5 times before finishing.\
        luong10: after 1/2 num train steps, we start halving the learning rate
          for 10 times before finishing.\
      """)

    parser.add_argument("--num_train_steps",
                        type=int, default=12000,
                        help="Num steps to train.")
    parser.add_argument("--colocate_gradients_with_ops",
                        type="bool", nargs="?",
                        const=True,
                        default=True,
                        help=("Whether try colocating gradients with "
                              "corresponding op"))

    # initializer
    parser.add_argument("--init_op",
                        type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight",
                        type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))

    # data
    parser.add_argument("--src",
                        type=str, default=None,
                        help="Source suffix, e.g., en.")
    parser.add_argument("--tgt",
                        type=str, default=None,
                        help="Target suffix, e.g., de.")
    parser.add_argument("--src_spk",
                        type=str, default=None,
                        help="Source speaker suffix, e.g., en.")
    parser.add_argument("--tgt_spk",
                        type=str, default=None,
                        help="Target speaker suffix, e.g., de.")
    parser.add_argument("--train_prefix",
                        type=str, default=None,
                        help="Train prefix, expect files with src/tgt "
                             "suffixes.")
    parser.add_argument("--dev_prefix",
                        type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix",
                        type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir",
                        type=str, default=None,
                        help="Store log/model files.")

    # Vocab
    parser.add_argument("--vocab_prefix",
                        type=str, default=None,
                        help="Vocab prefix, expect files with src/tgt "
                             "suffixes.")

    parser.add_argument("--embed_prefix",
                        type=str, default=None,
                        help="Pretrained embedding prefix, expect files "
                             "with src/tgt suffixes. The embedding files "
                             "should be Glove formated txt files.")
    parser.add_argument("--sos",
                        type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos",
                        type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--share_vocab",
                        type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to use the source vocab and embeddings "
                             "for both source and target.")
    parser.add_argument("--check_special_token",
                        type="bool", default=True,
                        help="Whether check special sos, eos, unk tokens "
                             "exist in the vocab files.")
    # Speaker


    parser.add_argument("--speaker_prefix",
                        type=str, default=None,
                        help="Speaker_file, no suffix added")

    parser.add_argument("--embed_speaker_prefix",
                        type=str, default=None,
                        help="Pretrained embedding for speakers"
                             "should be Glove formated txt files.")

    # Sequence lengths
    parser.add_argument("--src_max_len",
                        type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len",
                        type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer",
                        type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer",
                        type=int, default=None,
                        help="Max length of tgt sequences during inference. "
                             "Also use to restrict the maximum decoding "
                             "length.")

    # Default settings works well (rarely need to change)
    parser.add_argument("--unit_type",
                        type=str, default="lstm",
                        help="lstm | gru | layer_norm_lstm | nas")
    parser.add_argument("--forget_bias",
                        type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout",
                        type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm",
                        type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--batch_size",
                        type=int, default=128, help="Batch size.")

    parser.add_argument("--steps_per_stats",
                        type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train",
                        type=int, default=0,
                        help="Limit on the size of training data (0: no "
                             "limit).")
    parser.add_argument("--num_buckets",
                        type=int, default=5,
                        help="Put data into similar-length buckets.")

    # SPM
    parser.add_argument("--subword_option",
                        type=str, default="",
                        choices=["", "bpe", "spm"],
                        help="Set to bpe or spm to activate subword "
                             "desegmentation.")

    # Misc
    parser.add_argument("--num_gpus",
                        type=int, default=1,
                        help="Number of gpus in each worker.")
    parser.add_argument("--log_device_placement",
                        type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--metrics",
                        type=str, default="bleu",
                        help=("Comma-separated list of evaluations "
                              "metrics (bleu,rouge,accuracy)"))
    parser.add_argument("--steps_per_external_eval",
                        type=int, default=None,
                        help="How many training steps to do per external "
                             "evaluation. Automatically set based on data if "
                             "None.")
    parser.add_argument("--scope",
                        type=str, default=None,
                        help="scope to put variables under")
    parser.add_argument("--hparams_path",
                        type=str, default=None,
                        help="Path to standard hparams json file that "
                             "overrides hparams values from FLAGS.")
    parser.add_argument("--random_seed",
                        type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--override_loaded_hparams",
                        type="bool", nargs="?",
                        const=True, default=False,
                        help="Override loaded hparams with values specified")
    parser.add_argument("--num_keep_ckpts",
                        type=int, default=5,
                        help="Max number of checkpoints to keep.")
    parser.add_argument("--avg_ckpts",
                        type="bool", nargs="?",
                        const=True, default=False,
                        help="Average the last N checkpoints for external "
                             "evaluation. N can be controlled by setting "
                             "--num_keep_ckpts.")

    # Inference
    parser.add_argument("--ckpt",
                        type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    parser.add_argument("--inference_input_file",
                        type=str, default=None,
                        help="Set to the text to decode.")
    parser.add_argument("--inference_list",
                        type=str, default=None,
                        help=("A comma-separated list of sentence indices "
                              "(0-based) to decode."))
    parser.add_argument("--infer_batch_size",
                        type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file",
                        type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file",
                        type=str, default=None,
                        help="Reference file to compute evaluation scores "
                             "(if provided).")
    parser.add_argument("--beam_width",
                        type=int, default=0,
                        help=("beam width when using beam search decoder. If "
                              "0 (default), use standard decoder with greedy "
                              "helper."))
    parser.add_argument("--length_penalty_weight",
                        type=float, default=0.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--sampling_temperature",
                        type=float,
                        default=0.0,
                        help=("Softmax sampling temperature for inference "
                              "decoding, 0.0 means greedy decoding. This "
                              "option is ignored when using beam search."))
    parser.add_argument("--num_translations_per_input",
                        type=int, default=1,
                        help=("Number of translations generated for each "
                              "sentence. This is only used for inference."))

    # Job info
    parser.add_argument("--jobid",
                        type=int, default=0,
                        help="Task id of the worker.")
    parser.add_argument("--num_workers",
                        type=int, default=1,
                        help="Number of workers (inference only).")
    parser.add_argument("--num_inter_threads",
                        type=int, default=0,
                        help="number of inter_op_parallelism_threads")
    parser.add_argument("--num_intra_threads",
                        type=int, default=0,
                        help="number of intra_op_parallelism_threads")


def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        src_spk=flags.src_spk,
        tgt_spk=flags.tgt_spk,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        embed_prefix=flags.embed_prefix,
        speaker_prefix=flags.speaker_prefix,
        embed_speaker_prefix=flags.embed_speaker_prefix,
        out_dir=flags.out_dir,

        # Networks
        num_units=flags.num_units,
        num_units_speaker=flags.num_units_speaker,
        num_layers=flags.num_layers,  # Compatible
        num_encoder_layers=(flags.num_encoder_layers or flags.num_layers),
        num_decoder_layers=(flags.num_decoder_layers or flags.num_layers),
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,
        num_embeddings_partitions=flags.num_embeddings_partitions,

        # Attention mechanisms
        attention=flags.attention,
        attention_architecture=flags.attention_architecture,
        output_attention=flags.output_attention,
        pass_hidden_state=flags.pass_hidden_state,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        init_op=flags.init_op,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        warmup_steps=flags.warmup_steps,
        warmup_scheme=flags.warmup_scheme,
        decay_scheme=flags.decay_scheme,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        sampling_temperature=flags.sampling_temperature,
        num_translations_per_input=flags.num_translations_per_input,

        # Vocab
        sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,
        subword_option=flags.subword_option,
        check_special_token=flags.check_special_token,

        # Misc
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        share_vocab=flags.share_vocab,
        metrics=flags.metrics.split(","),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
        override_loaded_hparams=flags.override_loaded_hparams,
        num_keep_ckpts=flags.num_keep_ckpts,
        avg_ckpts=flags.avg_ckpts,
        num_intra_threads=flags.num_intra_threads,
        num_inter_threads=flags.num_inter_threads,
        )


def extend_hparams(hparams):
    """Extend training hparams."""
    assert hparams.num_encoder_layers and hparams.num_decoder_layers
    if hparams.num_encoder_layers != hparams.num_decoder_layers:
        hparams.pass_hidden_state = False
        utils.print_out(
            "Num encoder layer %d is different from num decoder layer"
            " %d, so set pass_hidden_state to False" % (
                hparams.num_encoder_layers,
                hparams.num_decoder_layers))
    print("##########\n\n\n##############")
    # Sanity checks
    if hparams.encoder_type == "bi" and hparams.num_encoder_layers % 2 != 0:
        raise ValueError("For bi, num_encoder_layers %d should be even" %
                         hparams.num_encoder_layers)
    if (hparams.attention_architecture in ["gnmt"] and
                hparams.num_encoder_layers < 2):
        raise ValueError("For gnmt attention architecture, "
                         "num_encoder_layers %d should be >= 2" %
                         hparams.num_encoder_layers)

    # Set residual layers
    num_encoder_residual_layers = 0
    num_decoder_residual_layers = 0
    if hparams.residual:
        if hparams.num_encoder_layers > 1:
            num_encoder_residual_layers = hparams.num_encoder_layers - 1
        if hparams.num_decoder_layers > 1:
            num_decoder_residual_layers = hparams.num_decoder_layers - 1

        if hparams.encoder_type == "gnmt":
            # The first unidirectional layer (after the bi-directional layer) in
            # the GNMT encoder can't have residual connection due to the
            # input is
            # the concatenation of fw_cell and bw_cell's outputs.
            num_encoder_residual_layers = hparams.num_encoder_layers - 2

            # Compatible for GNMT models
            if hparams.num_encoder_layers == hparams.num_decoder_layers:
                num_decoder_residual_layers = num_encoder_residual_layers
    hparams.add_hparam("num_encoder_residual_layers",
                       num_encoder_residual_layers)
    hparams.add_hparam("num_decoder_residual_layers",
                       num_decoder_residual_layers)

    if hparams.subword_option and hparams.subword_option not in ["spm", "bpe"]:
        raise ValueError("subword option must be either spm, or bpe")

    # Flags
    utils.print_out("# hparams:")
    utils.print_out("  src=%s" % hparams.src)
    utils.print_out("  tgt=%s" % hparams.tgt)
    utils.print_out("  train_prefix=%s" % hparams.train_prefix)
    utils.print_out("  dev_prefix=%s" % hparams.dev_prefix)
    utils.print_out("  test_prefix=%s" % hparams.test_prefix)
    utils.print_out("  out_dir=%s" % hparams.out_dir)

    ## Vocab
    # Get vocab file names first
    if hparams.vocab_prefix:
        src_vocab_file = hparams.vocab_prefix + "." + hparams.src
        tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt
    else:
        raise ValueError("hparams.vocab_prefix must be provided.")

    # Source vocab
    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
        src_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)

    # Target vocab
    if hparams.share_vocab:
        utils.print_out("  using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
            tgt_vocab_file,
            hparams.out_dir,
            check_special_token=hparams.check_special_token,
            sos=hparams.sos,
            eos=hparams.eos,
            unk=vocab_utils.UNK)
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
    hparams.add_hparam("src_vocab_file", src_vocab_file)
    hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

    # Pretrained Embeddings:
    hparams.add_hparam("src_embed_file", "")
    hparams.add_hparam("tgt_embed_file", "")
    if hparams.embed_prefix:
        src_embed_file = hparams.embed_prefix + "." + hparams.src
        tgt_embed_file = hparams.embed_prefix + "." + hparams.tgt

        if tf.gfile.Exists(src_embed_file):
            hparams.src_embed_file = src_embed_file

        if tf.gfile.Exists(tgt_embed_file):
            hparams.tgt_embed_file = tgt_embed_file

    ## Speakers
    # Get vocab file names first
    if hparams.speaker_prefix:
        speaker_file = hparams.speaker_prefix
    else:
        raise ValueError("hparams.speaker_file must be provided.")

    # Source vocab
    speaker_size, speaker_file = vocab_utils.check_vocab(
        speaker_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)

    hparams.add_hparam("speaker_size", speaker_size)
    hparams.add_hparam("speaker_file", speaker_file)

    # Pretrained Embeddings:
    hparams.add_hparam("spk_embed_file", "")
    if hparams.embed_speaker_prefix:
        spk_embed_file = hparams.embed_speaker_prefix

        if tf.gfile.Exists(spk_embed_file):
            hparams.spk_embed_file = spk_embed_file


    # Check out_dir
    if not tf.gfile.Exists(hparams.out_dir):
        utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
        tf.gfile.MakeDirs(hparams.out_dir)

    # Evaluation
    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

        if hparams.avg_ckpts:
            hparams.add_hparam("avg_best_" + metric, 0)  # larger is better
            best_metric_dir = os.path.join(hparams.out_dir,
                                           "avg_best_" + metric)
            hparams.add_hparam("avg_best_" + metric + "_dir", best_metric_dir)
            tf.gfile.MakeDirs(best_metric_dir)

    return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
    """Make sure the loaded hparams is compatible with new changes."""
    default_hparams = utils.maybe_parse_standard_hparams(
        default_hparams, hparams_path)

    # For compatible reason, if there are new fields in default_hparams,
    #   we add them to the current hparams
    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

    # Update all hparams' keys if override_loaded_hparams=True
    if default_hparams.override_loaded_hparams:
        for key in default_config:
            if getattr(hparams, key) != default_config[key]:
                utils.print_out("# Updating hparams.%s: %s -> %s" %
                                (key, str(getattr(hparams, key)),
                                 str(default_config[key])))
                setattr(hparams, key, default_config[key])
    return hparams


def create_or_load_hparams(
        out_dir, default_hparams, hparams_path, save_hparams=True):
    """Create hparams or load hparams from out_dir."""
    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(
            hparams, hparams_path)
        hparams = extend_hparams(hparams)
    else:
        hparams = ensure_compatible_hparams(hparams, default_hparams,
                                            hparams_path)

    # # Save HParams
    # if save_hparams:
    #     utils.save_hparams(out_dir, hparams)
    #     for metric in hparams.metrics:
    #         utils.save_hparams(os.path.join(out_dir, "best_" + metric + "_dir"),
    #                            hparams)

    # Print HParams
    utils.print_hparams(hparams)
    return hparams


def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
    """Run main."""
    # Job
    jobid = flags.jobid
    num_workers = flags.num_workers
    utils.print_out("# Job id %d" % jobid)

    # Random
    random_seed = flags.random_seed
    if random_seed is not None and random_seed > 0:
        utils.print_out("# Set random seed to %d" % random_seed)
        random.seed(random_seed + jobid)
        np.random.seed(random_seed + jobid)

    ## Train / Decode
    out_dir = flags.out_dir
    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)

    # Load hparams.
    hparams = create_or_load_hparams(
        out_dir, default_hparams, flags.hparams_path, save_hparams=(jobid == 0))

    if flags.inference_input_file:
        # Inference indices
        hparams.inference_indices = None
        if flags.inference_list:
            (hparams.inference_indices) = (
                [int(token) for token in flags.inference_list.split(",")])

        # Inference
        trans_file = flags.inference_output_file
        ckpt = flags.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        inference_fn(ckpt, flags.inference_input_file,
                     trans_file, hparams, num_workers, jobid)

        # Evaluation
        ref_file = flags.inference_ref_file
        if ref_file and tf.gfile.Exists(trans_file):
            for metric in hparams.metrics:
                score = evaluation_utils.evaluate(
                    ref_file,
                    trans_file,
                    metric,
                    hparams.subword_option)
                utils.print_out("  %s: %.1f" % (metric, score))
    else:
        # Train
        train_fn(hparams, target_session=target_session)


def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    train_fn = train.train
    inference_fn = inference.inference
    run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
