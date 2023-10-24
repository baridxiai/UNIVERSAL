# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import os
from UNIVERSAL.basic_optimizer import learning_rate_op

cwd = os.getcwd()

class iTensorBoard(tf.keras.callbacks.TensorBoard):
    def on_train_batch_end(self, batch, logs=None):
        with self._train_writer.as_default():
            step =self.model.optimizer.iterations.read_value()
            lr_schedule = getattr(self.model.optimizer, "lr", None)
            if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr_schedule(step)
            elif lr_schedule is not None:
                lr = lr_schedule
            if step % self.update_freq == 0:
                for m in self.model.metrics:
                    tf.summary.scalar(m.name, m.result(),step=step)
                for v,k in logs.items():
                    tf.summary.scalar(v, k,step=step)
            tf.summary.scalar("lr", lr,step=step)
        return super(iTensorBoard, self).on_train_batch_end(batch, logs=logs)

def get_callbacks(parameters,model_path):
    TFboard = iTensorBoard(
        log_dir=cwd + "/model_summary/",  update_freq=200
    )
    TFchechpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path + "/model_checkpoint" + "/model.{epoch:02d}-{loss:.2f}.ckpt",
        monitor="loss" ,
        save_weights_only=True,
        save_freq=parameters["callback_save_freq"],
        verbose=1,
        save_best_only=parameters["callback_save_best"],
    )
    NaNchecker = tf.keras.callbacks.TerminateOnNaN()
    backup = tf.keras.callbacks.BackupAndRestore(backup_dir=model_path + "/model_backup")
    call_backs = [
        backup,
        TFboard,
        TFchechpoint,
        NaNchecker,
        # LRTensorBoard
    ]
    return call_backs
