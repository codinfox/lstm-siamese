import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import numpy as np
from utils import maybe_download as maybe_download

num_hidden = 50
num_layers = 1

class siamese:

    def __init__(self, num_features):
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        self.x1 = tf.placeholder(tf.float32, [None, None, num_features])
        self.x2 = tf.placeholder(tf.float32, [None, None, num_features])
        # 1d array of size [batch_size]
        self.seq_len1 = tf.placeholder(tf.int32, [None])
        self.seq_len2 = tf.placeholder(tf.int32, [None])

        with tf.variable_scope("RNN") as scope:
            self.o1 = self.network(self.x1, self.seq_len1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2, self.seq_len2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()

    def network(self, inputs, seq_len):
        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers,
                                            state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, hidden_units = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        return hidden_units # TODO: how to correctly extract features

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.sub(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.sub(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.mul(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.sub(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.mul(labels_f, tf.maximum(0.0, tf.sub(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.mul(labels_f, tf.pow(tf.maximum(tf.sub(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

num_features = 13 #placeholder
initial_learning_rate = 1e-2

model = siamese(num_features)
train_step = tf.train.MomentumOptimizer(initial_learning_rate, 0.9).minimize(model.loss)

# Loading the data

audio_filename = maybe_download('LDC93S1.wav', 93638)
target_filename = maybe_download('LDC93S1.txt', 62)

fs, audio = wav.read(audio_filename)

inputs = mfcc(audio, samplerate=fs)
# Tranform in 3D array
train_inputs = np.asarray(inputs[np.newaxis, :])
train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)
train_seq_len = [train_inputs.shape[1]]

with tf.Session() as sess:
    # Initializate the weights and biases
    tf.initialize_all_variables().run()

    for step in range(100000):
        batch_x1, batch_y1, seq_len1 = train_inputs, np.array([1]), train_seq_len
        batch_x2, batch_y2, seq_len2 = train_inputs, np.array([1]), train_seq_len
        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, model.loss], feed_dict={
                            model.x1: batch_x1, 
                            model.x2: batch_x2, 
                            model.seq_len1: seq_len1,
                            model.seq_len2: seq_len2,
                            model.y_: batch_y})

        print loss_v
