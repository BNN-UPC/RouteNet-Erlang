import os
import numpy as np
import sys
sys.path.insert(1, '../')

from read_dataset import input_fn
from model import GNN_Model
import configparser
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def transformation(x, y):
    traffic_mean = 1650.59814453125
    traffic_std = 855.7061767578125
    packets_mean = 1.650602102279663
    packets_std = 0.8556720614433289
    capacity_mean = 25457.9453125
    capacity_std = 16221.1337890625

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std

    x["packets"] = (x["packets"] - packets_mean) / packets_std

    x["capacity"] = (x["capacity"] - capacity_mean) / capacity_std

    return x, y

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

ds_train = input_fn('../../data/scheduling/train', label='drops', shuffle=True)
ds_train = ds_train.map(lambda x, y: transformation(x, y))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_test = input_fn('../../data/scheduling/test', label='drops', shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['HYPERPARAMETERS']['learning_rate']))

model = GNN_Model(params)

loss_object = tf.keras.losses.MeanAbsoluteError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=['MAE'])

ckpt_dir = 'ckpt_dir'
latest = tf.train.latest_checkpoint(ckpt_dir)

if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_MAE:.3f}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    monitor='val_non_zero_MAPE',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

model.fit(ds_train,
          epochs=200,
          steps_per_epoch=4000,
          validation_data=ds_test,
          validation_steps=1000,
          callbacks=[cp_callback],
          use_multiprocessing=True)

best = tf.train.latest_checkpoint(ckpt_dir)
model.load_weights(best)

predictions = model.predict(ds_test)
with open('predictions.npy', 'wb') as f:
    np.save(f, predictions)
