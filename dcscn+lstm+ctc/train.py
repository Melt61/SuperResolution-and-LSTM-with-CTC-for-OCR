import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
import random
import time
import datetime

from PIL import Image

from tools import DataLoader

#learning parameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.99
epochs = 100
batch_size = 32

validation_steps = 500

#network parameters
input_size = 50
hidden_neuron = 512
layer_num = 2
#num_classes = 96
num_classes = 5689
fc_hidden_neuron = 256

#loading data 
image_path = '/home/melt61/PictureGenerator/GenImage07'
label_path = '/home/melt61/PictureGenerator/GenLabel07/labels.txt'

valid_image_path = '/home/melt61/PictureGenerator/GenImage08'
valid_label_path = '/home/melt61/PictureGenerator/GenLabel08/labels.txt'

input_data_loader = DataLoader.data_loader(image_path, label_path)
valid_input_data_loader = DataLoader.data_loader(valid_image_path, valid_label_path)

num_epochs = epochs
num_train_samples = input_data_loader.get_input_len()
num_val_samples = valid_input_data_loader.get_input_len()
num_batches_per_epoch = num_train_samples//num_epochs
num_batches_per_epoch_val = num_val_samples//num_epochs
shuffle_idx_val = np.random.permutation(num_val_samples)

print("train samples: ",num_train_samples,"  valid samples: ",num_val_samples)

#define graph
graph_1 = tf.Graph()
with graph_1.as_default():
    #input
    inputs = tf.placeholder(tf.float32,[batch_size, None, input_size])
    labels = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32,[None])
    
    #build LSTM
    #stack = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_neuron, state_is_tuple = True) 
    #                                     for i in range(layer_num)], 
    #                                    state_is_tuple = True)

    #stack_back = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_neuron, state_is_tuple = True) 
    #                                     for i in range(layer_num)], 
    #                                    state_is_tuple = True)

    rnn_cells_fw = [tf.contrib.rnn.LSTMCell(hidden_neuron, state_is_tuple = True) for i in range(layer_num)]
    rnn_cells_bw = [tf.contrib.rnn.LSTMCell(hidden_neuron, state_is_tuple = True) for i in range(layer_num)]

    outputs, _ , _= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw = rnn_cells_fw, 
                                                                cells_bw = rnn_cells_bw, 
                                                                inputs = inputs, sequence_length = seq_len, dtype = tf.float32)
    
    #classcification process
    in_shape = tf.shape(inputs)
    batch_s, max_timesteps = in_shape[0], in_shape[1]
    outputs = tf.reshape(outputs, [-1, hidden_neuron*2])

    #wh = tf.get_variable(name = 'wh',
    #                    shape = [hidden_neuron, fc_hidden_neuron],
    #                    dtype = tf.float32,
    #                    initializer = tf.contrib.layers.xavier_initializer())

    #bh = tf.get_variable(name = 'bh',
    #                    shape = [fc_hidden_neuron],
    #                    dtype = tf.float32,
    #                    initializer = tf.constant_initializer())

    w = tf.get_variable(name = 'w',
                       shape = [hidden_neuron*2, num_classes],
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

    predict_accuracy = 1-acc_rate

    tf.summary.scalar('predict accuracy', predict_accuracy)

    merged_summary = tf.summary.merge_all()
    

#training
with tf.device('/cpu:0'):
    config = tf.ConfigProto(allow_soft_placement = True)
    with tf.Session(config = config,graph=graph_1) as sess:
        #initialize
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('./logs/chi_5-10', sess.graph)
        sess.run(tf.global_variables_initializer()) 
        
        print('=============================begin training=============================')
        for cur_epoch in range(num_epochs):
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()

                # the tracing part
            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()
                indexs = [shuffle_idx[i % num_train_samples] for i in
                            range(cur_batch * batch_size, (cur_batch + 1) * batch_size)]
                batch_inputs, batch_seq_len, batch_labels = \
                        input_data_loader.get_batch_by_index(indexs)

                feed = {inputs: batch_inputs,
                        labels: batch_labels,
                        seq_len: batch_seq_len} 
                #print(batch_inputs)
                #print(batch_labels)
                #print(batch_seq_len)
                    # if summary is needed
                    # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)

                summary_str, batch_cost, step, train_acc,_= \
                        sess.run([merged_summary, cost, global_step, acc_rate,optimizer
                                ], feed)
                    # calculate the cost
                train_cost += batch_cost * batch_size
                #if (cur_batch + 1) % 100 == 0:
                print('batch', cur_batch, ': time', time.time() - batch_time,'   train_accuracy: ',1-train_acc)
                summary_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    #if step % FLAGS.save_steps == 1:
                    #    if not os.path.isdir(FLAGS.checkpoint_dir):
                    #        os.mkdir(FLAGS.checkpoint_dir)
                    #    logger.info('save the checkpoint of{0}', format(step))
                    #    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                    #               global_step=step)

                    # train_err += the_err * FLAGS.batch_size
                    # do validation
                if step % validation_steps == 0:
                    saver.save(sess, "./model/model_04(chi_5-10)/model_03.ckpt")
                    acc_batch_total = 0
                    ler = 0
                    lr = 0
                    for j in range(num_batches_per_epoch_val):
                        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                        range(j * batch_size, (j + 1) * batch_size)]
                        val_inputs, val_seq_len, val_labels = \
                                valid_input_data_loader.get_batch_by_index(indexs_val)
                        val_feed = {inputs: val_inputs,
                                    labels: val_labels,
                                    seq_len: val_seq_len}

                        dense_de, acc= \
                                sess.run([dense_decoded, acc_rate],
                                        val_feed)

                        #aver_label_length = val_labels[0].shape[0]/val_labels[2][0]

                        #print(int(batch_labels[0].shape[0]),'  and   ',int(batch_labels[2].shape[0]))

                        #print('aver= ',aver_label_length)

                        #acc_rate = 1-aed/aver_label_length
                            # print the decode result
                        #ri_labels = valid_input_data_loader.the_label(indexs_val)
                        #acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                            #ignore_value=-1, isPrint=True)
                        acc_batch_total = acc_batch_total+(1-acc)

                    accuracy = acc_batch_total / num_batches_per_epoch_val

                    avg_train_cost = train_cost / ((cur_batch + 1) * batch_size)

                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                        "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                        "time = {:.3f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                    cur_epoch + 1, epochs, accuracy, avg_train_cost,
                                    time.time()-start_time))

        
