import os
import numpy as np
import re
from read_dataset import input_fn
import configparser
import tensorflow as tf

METRIC = 'Delay'
import sys
sys.path.insert(1, './{}'.format(METRIC))
from model import GNN_Model
TRAFFIC_MODEL = 'K1'
MODEL_CKPT_DIR = './{}/{}'.format(METRIC,TRAFFIC_MODEL)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def transformation(x, y):
    traffic_mean = 660.5723876953125
    traffic_std = 420.22003173828125
    packets_mean = 0.6605737209320068
    packets_std = 0.42021000385284424
    capacity_mean = 25442.669921875
    capacity_std = 16217.9072265625

    x["traffic"] = (x["traffic"] - traffic_mean) / traffic_std

    x["packets"] = (x["packets"] - packets_mean) / packets_std

    x["capacity"] = (x["capacity"] - capacity_mean) / capacity_std

    return x, tf.math.log(y)

def denorm_MAPE(y_true, y_pred):
    denorm_y_true = tf.math.exp(y_true)
    denorm_y_pred = tf.math.exp(y_pred)
    return tf.abs((denorm_y_pred - denorm_y_true) / denorm_y_true) * 100


params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read(os.path.join(MODEL_CKPT_DIR,'config.ini'))

ds_test = input_fn('../data/time-dist-experiments/Deterministic/test', label=METRIC.lower(), shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['HYPERPARAMETERS']['learning_rate']))

model = GNN_Model(params)

loss_object = tf.keras.losses.MeanSquaredError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=denorm_MAPE)
model(input_fn('../data/time-dist-experiments/Deterministic/test', label=METRIC.lower(), shuffle=False).take(1))
best = None
best_mre = float('inf')
ckp_dir = os.path.join(MODEL_CKPT_DIR,'ckpt_dir')
for f in os.listdir(ckp_dir):
    if os.path.isfile(os.path.join(ckp_dir, f)):
        reg = re.findall("\d+\.\d+", f)
        if len(reg) > 0:
            mre = float(reg[0])
            if mre <= best_mre:
                best = f.replace('.index', '')
                best_mre = mre

print("BEST CHECKOINT FOUND: {}".format(best))
model.load_weights(os.path.join(ckp_dir), best)

predictions = model.predict(ds_test)
predictions = np.exp(predictions)
np.save(os.path.join(MODEL_CKPT_DIR,'predictions.npy', predictions))

