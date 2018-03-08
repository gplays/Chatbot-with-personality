
from __future__ import print_function

import math
import os
import random
import time

import tensorflow as tf

from .. import attention_model
from .. import gnmt_model
from .. import inference
from .. import model as nmt_model
from .. import model_helper
from ..utils import misc_utils as utils
from ..utils import nmt_utils
from .train import init_stats
update_stats
print_step_info
process_stats

utils.check_tensorflow_version()

__all__ = [
    "run_sample_decode", "run_internal_eval", "run_external_eval",
    "run_avg_external_eval", "run_full_eval", "init_stats", "update_stats",
    "print_step_info", "process_stats", "train"
    ]

def train_persona(hparams, scope=None, target_session=""):
    "Train a Persona guessing Model"

    out_dir = lambda x: os.path.join(hparams.out_dir, x)
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    steps_per_eval = 10 * steps_per_stats

    model_creator = attention_model.AttentionModel

    # No inference model because sample decoding is meaningless and we have a
    #  single metrics so no external eval
    train_model = model_helper.create_train_model_personna(model_creator,
                                                  hparams, scope)
    eval_model = model_helper.create_eval_model_personna(model_creator,
                                                         hparams, scope)

    summary_name = "train_persona_log"
    model_dir = hparams.out_dir

    # Log and output files
    log_file = out_dir("log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("# log_file=%s" % log_file, log_f)

    # TensorFlow model
    config_proto = utils.get_config_proto(hparams=hparams)

    train_sess = tf.Session(target_session, train_model.graph, config_proto)
    eval_sess = tf.Session(target_session, eval_model.graph, config_proto, )

    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(
            train_model.model, model_dir, train_sess, "train")

    # Summary writer
    summary_path = out_dir(summary_name)
    summary_writer = tf.summary.FileWriter(summary_path, train_model.graph)

    last_stats_step = global_step
    last_eval_step = global_step

    # This is the training loop.
    stats, info, start_train_time = before_train(
        loaded_train_model, train_model, train_sess, global_step, hparams,
        log_f)

    while global_step < num_train_steps:
        ### Run a step ###
        start_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            hparams.epoch_step = 0
            utils.print_out(
                "# Finished an epoch, step %d. Perform external evaluation" %
                global_step)
            run_internal_eval_personna(eval_model, eval_sess, model_dir,
                                       hparams,
                              summary_writer)

            train_sess.run(train_model.iterator.initializer,
                           feed_dict={train_model.skip_count_placeholder: 0})
            continue

        # Process step_result, accumulate stats, and write summary
        global_step, info["learning_rate"], step_summary = update_stats(
            stats, start_time, step_result)
        summary_writer.add_summary(step_summary, global_step)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step
            try:
                process_stats(stats, info, global_step, steps_per_stats, log_f)
            except OverflowError:
                break
            finally:
                print_step_info("  ", global_step, info,
                                _get_best_results(hparams), log_f)

            # Reset statistics
            stats = init_stats()

        if global_step - last_eval_step >= steps_per_eval:
            last_eval_step = global_step
            print_eval(global_step, summary_writer, info)

        # Save checkpoint
        loaded_train_model.saver.save(train_sess,
                                      out_dir("translate.ckpt"),
                                      global_step=global_step)
        run_internal_eval_personna(eval_model, eval_sess, model_dir,
                                   hparams, summary_writer)

    # Done training
    loaded_train_model.saver.save(
        train_sess,
        out_dir( "translate.ckpt"),
        global_step=global_step)

    #TODO modify the following call to account for no external metrics
    result_summary, _, final_eval_metrics = run_full_eval_persona(
            model_dir, eval_model, eval_sess, hparams,summary_writer)
    print_step_info("# Final, ", global_step, info, result_summary, log_f)
    utils.print_time("# Done training!", start_train_time)

    summary_writer.close()

    #TODO understand the following snippet: they were already evaluated, right?
    utils.print_out("# Start evaluating saved best models.")
    for metric in hparams.metrics:
        best_model_dir = getattr(hparams, "best_" + metric + "_dir")
        #changed infer_model to eval_model
        summary_writer = tf.summary.FileWriter(
            os.path.join(best_model_dir, summary_name), eval_model.graph)
        result_summary, best_global_step, _ = run_full_eval_persona(
            best_model_dir, eval_model, eval_sess,hparams,summary_writer)

        print_step_info("# Best %s, " % metric, best_global_step, info,
                        result_summary, log_f)
        summary_writer.close()

    #skipping part about avg ckpt

    return final_eval_metrics, global_step