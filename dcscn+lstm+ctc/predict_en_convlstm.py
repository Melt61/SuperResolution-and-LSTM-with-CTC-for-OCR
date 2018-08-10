import tensorflow as tf
#from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
import random
import time
import datetime

from PIL import Image

from tools import DataLoader
from tools import CharactorSource




input_image_path = '/home/melt61/PictureGenerator/GenImage10/423.jpg'
input_image = Image.open(input_image_path).convert('L')
input_image.show()
input_image = np.asarray(input_image)
input_image = input_image.transpose(1, 0)

input_length = input_image.shape[0]
input_image = input_image.reshape(input_length, 50, 1)

network_input = []
network_input.append(input_image)



input_seq_len = []
input_seq_len.append(input_image.shape[0])




#learning parameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.99
epochs = 100
batch_size = 1

validation_steps = 500

#network parameters
input_size = 50
hidden_neuron = 512
layer_num = 2
num_classes = 96
#num_classes = 5689
fc_hidden_neuron = 256

#define graph
graph_1 = tf.Graph()
with graph_1.as_default():
    #input
    inputs = tf.placeholder(tf.float32,[batch_size, None, input_size]+[1])
    labels = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32,[None])
    

    in_shape = tf.shape(inputs)
    batch_s, max_timesteps = in_shape[0], in_shape[1]
    #build LSTM
    rnn_cell_layer_1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=1, input_shape=[50, 1], output_channels=32, kernel_shape=[3], name='convlstm_layer_01')

    #initial_state = rnn_cells.zero_state(batch_size, dtype=tf.float32)

    #stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

    output_1, _= tf.nn.dynamic_rnn(rnn_cell_layer_1, inputs, dtype=tf.float32, scope='convlstm_layer_01')

    rnn_cell_layer_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=1, input_shape=[50, 32], output_channels=64, kernel_shape=[3], name='convlstm_layer_02')

    output_2, _= tf.nn.dynamic_rnn(rnn_cell_layer_2, output_1, dtype=tf.float32, scope='convlstm_layer_02')

    rnn_cell_layer_3 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=1, input_shape=[50, 64], output_channels=128, kernel_shape=[3], name='convlstm_layer_03')

    #stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell_layer_1, rnn_cell_layer_2, rnn_cell_layer_3])

    outputs, _= tf.nn.dynamic_rnn(rnn_cell_layer_3, output_2, dtype=tf.float32, scope='convlstm_layer_03')

    #outputs, _ , _= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw = rnn_cells_fw, 
    #                                                            cells_bw = rnn_cells_bw, 
    #                                                            inputs = inputs, sequence_length = seq_len, dtype = tf.float32)
    
    #classcification process
    
    outputs = tf.reshape(outputs, [-1, 50*128])
    #wh = tf.get_variable(name = 'wh',
    #                    shape = [hidden_neuron, fc_hidden_neuron],
    #                    dtype = tf.float32,
    #                    initializer = tf.contrib.layers.xavier_initializer())

    #bh = tf.get_variable(name = 'bh',
    #                    shape = [fc_hidden_neuron],
    #                    dtype = tf.float32,
    #                    initializer = tf.constant_initializer())

    w = tf.get_variable(name = 'w',
                       shape = [50*128, num_classes],
                       dtype = tf.float32,
                       initializer = tf.contrib.layers.xavier_initializer())
    
    b = tf.get_variable(name = 'b',
                       shape = [num_classes],
                       dtype = tf.float32,
                       initializer = tf.constant_initializer())

    
    #logits = tf.matmul(outputs, wh) + bh

    #logits = tf.nn.tanh(logits)

    logits = tf.matmul(outputs, w) + b

    logits = tf.reshape(logits, [batch_s, -1, num_classes])
    logits = tf.transpose(logits, [1, 0, 2])
    
    #ctc loss
    global_step = tf.Variable(0, trainable = False)
    
    loss = tf.nn.ctc_loss(labels = labels, inputs = logits, sequence_length = seq_len)
    
    cost = tf.reduce_mean(loss)

    tf.summary.scalar('cost', cost)
    
    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                      beta1 = beta1,
                                      beta2 = beta2).minimize(loss,
                                                              global_step = global_step)
    
    #ctc decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated = False)
    
    dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value = -1)
    
    #error rate
    acc_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))


    merged_summay = tf.summary.merge_all()


with tf.device('/cpu:0'):
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config,graph=graph_1) as sess:
         saver = tf.train.Saver()
         saver.restore(sess, './model/model_05(convlstm_eng)/model_05.ckpt')

         feeds = {
             inputs : network_input,
             seq_len : input_seq_len
         } 
         predict_result = sess.run([dense_decoded], feeds)
         print("outputs:  ", predict_result)

         ens_ins =CharactorSource.charactorsource()
         predict_characters = []
         for i in predict_result[0]:
             eng_char = ens_ins.eng_int2char(i)
             predict_characters.append(eng_char)

         print("predict:  ",predict_characters)
