import sys
import re
sys.path.insert(1, "./code")
from link_division_model import LinkDivModel
import configparser
import tensorflow as tf
from link_division_dataset import input_fn

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

optimizer = tf.keras.optimizers.Adam()

model = LinkDivModel(params)

loss_object = tf.keras.losses.MeanAbsolutePercentageError()
model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False)
"""def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100

loss_object = tf.keras.losses.MeanSquaredError()
model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=denorm_MAPE)"""

best = None
best_mre = float('inf')
for f in os.listdir('./ckpt_dir'):
    if os.path.isfile(os.path.join('./ckpt_dir', f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = '.'.join(f.split('.')[:-1])
                best_mre = mre

print("BEST CHECKOINT FOUND: {}".format(best))
model.load_weights('./ckpt_dir/{}'.format(best))

# for i in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260,
#          280, 300]:
for i in [300]:
    print(f"TOPOLOGY SIZE: {i}")
    ds_test = input_fn(f'../data/scalability/test/{i}', min_scale=10, max_scale=11, shuffle=False)
    ds_test = ds_test.map(lambda x, y: transformation(x, y))
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    for s in range(10):
        ind_ds = ds_test.skip(s)
        model.evaluate(ind_ds, steps=1)
