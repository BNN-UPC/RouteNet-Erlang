import sys

sys.path.insert(1, "./code")
from link_division_model import LinkDivModel
import configparser
import tensorflow as tf
from link_division_dataset import input_fn

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

TRAINING_SIZES = list(range(10, 51, 5))
TRAINING_INTESITIES = list(range(0, 1))

TEST_SIZES = [250]
TEST_INTESITIES = list(range(0, 1))

def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100

def transformation(x, y):
    traffic_mean = 666.4519976306121
    traffic_std = 418.79412331425846
    packets_mean = 0.660199595571597
    packets_std = 0.4204438794894145
    bandwidth_mean = 21166.35
    bandwidth_std = 24631.01
    scale_mean = 10.5
    scale_std = 5.77

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std

    x["packets"] = (x["packets"] - packets_mean) / packets_std

    x["capacity"] = (x["capacity"] - bandwidth_mean) / bandwidth_std

    x["scale"] = (x["scale"] - scale_mean) / scale_std

    return x, y #tf.math.log(y)


def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100


params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

min_scale = 1
max_scale = 20
ds_train = input_fn('./data/train/', min_scale=min_scale, max_scale=max_scale, samples_per_sample=1, shuffle=True)
ds_train = ds_train.map(lambda x, y: transformation(x, y))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()
ds_train = ds_train.shuffle(1000)

# ds_test = load_snapshot(TEST_SIZES, TEST_INTESITIES, mode='validation')
ds_test = input_fn('./data/validation', min_scale=10, max_scale=11, shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


optimizer = tf.keras.optimizers.Adam()

model = LinkDivModel(params)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=['MAPE'])

ckpt_dir = params['DIRECTORIES']['logs'] + '/delay'
latest = tf.train.latest_checkpoint(ckpt_dir)

if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_MAPE:.2f}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    monitor='val_MAPE',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

model.fit(ds_train,
          epochs=100,
          steps_per_epoch=4000,
          validation_data=ds_test,
          callbacks=[cp_callback],
          batch_size=32,
          use_multiprocessing=True)

best = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(best)

model.evaluate(ds_test)