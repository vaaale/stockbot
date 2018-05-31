from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import pandas as pd

num_periods = 15
f_horizon = 1

num_input = 1
state_size = 50
num_classes = 3
output = 1
dropout = 0.5
num_layers = 2

num_epochs = 50
batch_size = 32
num_batches = 300
learning_rate = 0.001

random.seed(111)

onehot = np.identity(3)
def generate_batch(batch_size, num_periods, f_horizon):
    nb_samples = batch_size * num_periods + f_horizon
    rng = pd.date_range(start='2000', periods=nb_samples, freq='M')
    ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()

    x_data = ts[:(len(ts) - (len(ts) % num_periods))]
    x_batches = x_data.values.reshape(-1, num_periods, 1)

    # y_data = ts[1:(len(ts) - (len(ts) % num_periods)) + f_horizon]
    # y_batches = y_data.values.reshape(-1, num_periods, 1)
    y_data = np.eye(num_classes)[np.random.choice(num_classes, batch_size * num_periods)]
    y_batches = y_data.reshape(-1, num_periods, num_classes)

    return x_batches, y_batches


x, y = generate_batch(batch_size, num_periods, f_horizon)

print(x.shape)
print(y.shape)
np.random.choice([0, 1, 2], 5, p=[0.5, 0.25, 0.25])

# Create model
def create_model(batchX, batchY, init_state):
    state_per_layer_list = tf.unstack(init_state, axis=0)
    rnn_tuple_state = tuple(
        [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
         for idx in range(num_layers)]
    )


    cells = [tf.contrib.rnn.BasicLSTMCell(num_units=state_size, activation=tf.nn.relu, state_is_tuple=True) for _ in range(num_layers)]
    # cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout) for cell in cells]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    states_series, current_state = tf.nn.dynamic_rnn(cell, batchX, initial_state=rnn_tuple_state)
    states_series = tf.reshape(states_series, [-1, state_size])

    stacked_rnn_output = tf.reshape(states_series, [-1, state_size])
    stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
    logits = tf.layers.dense(stacked_outputs, num_classes)

    outputs = tf.arg_max(tf.nn.softmax(logits), 1)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batchY)
    total_loss = tf.reduce_mean(loss)
    # outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

    # loss = tf.reduce_sum(tf.square(outputs - batchY))


    return current_state, loss, total_loss, logits, outputs


# Define placeholders
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, None, num_input], name='PL_X')
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, None, num_classes], name='PL_Y')
init_state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size], name='PL_init_state')

# Build model
current_state, loss, total_loss, logits, outputs = create_model(batchX_placeholder, batchY_placeholder, init_state_placeholder)

# Build training step
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        print("Epoch", epoch_idx)

        for batch_idx in range(num_batches):
            _current_state = np.zeros((num_layers, 2, batch_size, state_size))
            batchX, batchY = generate_batch(batch_size, num_periods, f_horizon)

            _total_loss, _train_step, _current_state = sess.run(
                [total_loss, train_step, current_state],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state_placeholder: _current_state
                })

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Loss", _total_loss)

        save_path = saver.save(sess, "models/model.ckpt")
        print("Model saved in path: %s" % save_path)

plt.ioff()
plt.show()
