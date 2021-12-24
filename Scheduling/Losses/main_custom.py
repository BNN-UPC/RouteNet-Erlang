import os
import sys

sys.path.insert(1, '../../data/')

from read_dataset import input_fn
from model_custom import GNN_Model
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


def R_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def prob_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred
    ))
    return loss


def non_zero_MAPE(y_true, y_pred):
    mask = tf.greater(y_true, 0)
    non_zero_y_true = tf.boolean_mask(y_true, mask)
    non_zero_y_pred = tf.boolean_mask(y_pred, mask)
    return tf.keras.metrics.mean_absolute_percentage_error(non_zero_y_true, non_zero_y_pred)


params = configparser.ConfigParser()
params._interpolation = configparser.ExtendedInterpolation()
params.read('config.ini')

# ds_train = load_snapshot(TRAINING_SIZES, TRAINING_INTESITIES, mode='training', buffer_size=5000)
ds_train = input_fn('../../data/sched_big_losses/train', label='PktsDrop', shuffle=True)
ds_train = ds_train.map(lambda x, y: transformation(x, y))
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.repeat()

# ds_test = load_snapshot(TEST_SIZES, TEST_INTESITIES, mode='validation')
ds_test = input_fn('../../data/sched_big_losses/test', label='PktsDrop', shuffle=False)
ds_test = ds_test.map(lambda x, y: transformation(x, y))
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['HYPERPARAMETERS']['learning_rate']))

model = GNN_Model(params)

model.compile(loss=prob_loss,
              optimizer=optimizer,
              run_eagerly=False,
              metrics=[non_zero_MAPE, R_squared])

ckpt_dir = 'ckpt_dir_custom'
latest = tf.train.latest_checkpoint(ckpt_dir)

if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_non_zero_MAPE:.2f}")

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

"""ds_test = ds_test.take(1)
model.evaluate(ds_test)

for x, y in ds_test:
    pass
prediction = model.predict(ds_test)
print(prediction)
print(y)"""
