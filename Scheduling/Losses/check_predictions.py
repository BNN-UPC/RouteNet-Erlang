import os
import sys

sys.path.insert(1, '../../data/')

from read_dataset import input_fn
from model import GNN_Model
import configparser
import tensorflow as tf
import re
import numpy as np

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


def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100


params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

ds_test = input_fn('../../data/gnnet_data_set_evaluation_delays', label='PktsDrop', shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.take(1)

optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['HYPERPARAMETERS']['learning_rate']))

model = GNN_Model(params)

loss_object = tf.keras.losses.MeanSquaredError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=denorm_MAPE)

best = None
best_mre = float('inf')
for f in os.listdir('./ckpt_dir_mae'):
    if os.path.isfile(os.path.join('./ckpt_dir_mae', f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = f.replace('.index', '')
                best_mre = mre

print("BEST CHECKOINT FOUND: {}".format(best))
model.load_weights('./ckpt_dir_mae/{}'.format(best))

predictions = model.predict(ds_test)
print(predictions)
for x, y in ds_test:
    print(y)
