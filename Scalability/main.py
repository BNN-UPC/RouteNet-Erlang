import sys
import re
sys.path.insert(1, "./code")
from model import GNN_Model
import configparser
import tensorflow as tf
from read_dataset import input_fn

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    return x, y

params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

min_scale = 1
max_scale = 20
ds_train = input_fn('../data/scalability/train/', min_scale=min_scale, max_scale=max_scale, samples_per_sample=1,
                    shuffle=True)
ds_train = ds_train.map(lambda x, y: transformation(x, y))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()
ds_train = ds_train.shuffle(1000)

ds_test = input_fn('../data/scalability/test/', min_scale=10, max_scale=11, shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


optimizer = tf.keras.optimizers.Adam()

model = GNN_Model(params)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()
model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False)

ckpt_dir = 'ckpt_dir'
latest = tf.train.latest_checkpoint(ckpt_dir)

if latest is not None:
    print("Found a pretrained model, restoring...")
    best = None
    best_mre = float('inf')
    for f in os.listdir('./ckpt_dir'):
        if os.path.isfile(os.path.join('./ckpt_dir', f)):
            reg = re.findall("\d+\.\d+", f)
            if len(reg) > 0:
                mre = float(reg[0])
                if mre <= best_mre:
                    best = f.replace('.index', '')
                    best = best.replace('.data-00000-of-00001', '')
                    best_mre = mre

    print("BEST CHECKOINT FOUND: {}".format(best))
    model.load_weights('./ckpt_dir/{}'.format(best))
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

model.fit(ds_train,
          epochs=200,
          steps_per_epoch=4000,
          validation_data=ds_test,
          validation_steps=20,
          callbacks=[cp_callback],
          batch_size=32,
          use_multiprocessing=True)
