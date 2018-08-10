import tensorflow as tf
#from tensorflow.python.ops import ctc_ops as ctc
import warpctc_tensorflow
import numpy as np
import random
import time
import datetime
from tensorflow.contrib import layers
from tensorflow.python.layers.core import Dense
from tensorflow.python.client import timeline

from PIL import Image

from tools import DataLoader
from tools import CharactorSource

#learning parameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.99
epochs = 100
batch_size = 32

validation_steps = 1000

#network parameters
input_size = 50
hidden_neuron = 512
layer_num = 2
#num_classes = 96
num_classes = 5689
fc_hidden_neuron = 2048

image_height = 50

charSource = CharactorSource.charactorsource()

#att para
START_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2
VOCAB_ATT = charSource.char_to_int_att
VOCAB_ATT_SIZE = len(VOCAB_ATT)

VOCAB_CTC = charSource.char_to_int
VOCAB_CTC_SIZE = len(VOCAB_CTC)

ATT_EMBED_DIM = 512
BATCH_SIZE = 32
RNN_UNITS = 256
TRAIN_STEP = 1000000
IMAGE_HEIGHT = 50
MAXIMUM__DECODE_ITERATIONS = None
DISPLAY_STEPS = 100
LOGS_PATH = 'logs_path'
CKPT_DIR = 'save_model'

#loading data 
image_path = '/home/melt61/PictureGenerator/GenImage12'
label_path = '/home/melt61/PictureGenerator/GenLabel12/labels.txt'

valid_image_path = '/home/melt61/PictureGenerator/GenImage13'
valid_label_path = '/home/melt61/PictureGenerator/GenLabel13/labels.txt'

input_data_loader = DataLoader.data_loader(image_path, label_path)
valid_input_data_loader = DataLoader.data_loader(valid_image_path, valid_label_path)

num_epochs = epochs
num_train_samples = input_data_loader.get_input_len()
num_val_samples = valid_input_data_loader.get_input_len()
num_batches_per_epoch = num_train_samples//batch_size
num_batches_per_epoch_val = num_val_samples//batch_size
shuffle_idx_val = np.random.permutation(num_val_samples)

print("train samples: ",num_train_samples,"  valid samples: ",num_val_samples)

#function

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
        dtype: type of data
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def att_decode(helper, rnn_features, scope, reuse=None):
    """
    Attention decode part
    :param helper: train or inference
    :param rnn_features: encoded features
    :param scope: name scope
    :param reuse: reuse or not
    :return: attention decode output
    """
    with tf.variable_scope(scope, reuse=reuse):
        if  attention_mode == 1:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=RNN_UNITS,
                                                                    memory=rnn_features)
        else:
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=RNN_UNITS,
                                                                        memory=rnn_features)

        cell = tf.contrib.rnn.GRUCell(num_units=RNN_UNITS)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                        attention_layer_size=RNN_UNITS,
                                                        output_attention=True)
        output_layer = Dense(units=VOCAB_ATT_SIZE)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=attn_cell, helper=helper,
            initial_state=attn_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
            output_layer=output_layer)

        att_outputs = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder, output_time_major=False,
            impute_finished=True, maximum_iterations=MAXIMUM__DECODE_ITERATIONS)

        return att_outputs

#define graph
graph_1 = tf.Graph()
with graph_1.as_default():
    #input
    inputs = tf.placeholder(tf.float32,[batch_size, 250, 50])
    labels = tf.sparse_placeholder(tf.int32)
    flatten_labels = tf.placeholder(tf.int32,[None])
    seq_len = tf.placeholder(tf.int32,[None])
    label_len = tf.placeholder(tf.int32,[None])
    
    # loss weights refrencehttps://arxiv.org/pdf/1609.06773v1.pdf
    ctc_loss_weights = 0
    att_loss_weights = 1 - ctc_loss_weights
    # choose attention mode 0 is "Bahdanau" Attention, 1 is "Luong" Attention
    attention_mode = 1

    # visualization path and model saved path
    logs_path = LOGS_PATH
    save_model_dir = CKPT_DIR

    # input image
    #input_image = tf.placeholder(tf.float32, shape=(None, image_height, None, 1), name='img_data')

    # attention part placeholder
    att_train_output = tf.placeholder(tf.int64, shape=[None, None], name='att_train_output')
    att_train_length = tf.placeholder(tf.int32, shape=[None], name='att_train_length')
    att_target_output = tf.placeholder(tf.int64, shape=[None, None], name='att_target_output')

    # ctc part placeholder
    #ctc_label = tf.sparse_placeholder(tf.int32, name='ctc_label')
    #ctc_feature_length = tf.placeholder(tf.int32, shape=[None], name='ctc_feature_length')

    x_image = tf.reshape(inputs, [-1,50,250,1])

    #build cnn
    #layer 1  1*50*250 -> 64*48*248
    convolution1 = layers.conv2d(inputs=x_image,
                                    num_outputs=64,
                                    kernel_size=[3, 3],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu)
    #max pooling 1 64*48*248 -> 64*24*124 
    pool1 = layers.max_pool2d(inputs=convolution1, kernel_size=[2, 2], stride=[2, 2])

    #layer 2 64*24*124 -> 128*22*122
    convolution2 = layers.conv2d(inputs=pool1,
                                    num_outputs=128,
                                    kernel_size=[3, 3],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu)
    #max pooling 2 128*22*122 -> 128*11*61
    pool2 = layers.max_pool2d(inputs=convolution2, kernel_size=[2, 2], stride=[2, 2])

    #layer 3 128*11*61 -> 256*9*59
    convolution3 = layers.conv2d(inputs=pool2,
                                    num_outputs=256,
                                    kernel_size=[3, 3],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu)

    #layer 4 256*9*59 -> 256*7*57
    convolution4 = layers.conv2d(inputs=convolution3,
                                    num_outputs=256,
                                    kernel_size=[3, 3],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu)
    #max pooling 3 256*7*57 -> 256*4*57
    pool3 = layers.max_pool2d(inputs=convolution4, kernel_size=[2, 1], stride=[2, 1])

    #layer 5 256*4*57 -> 512*2*55
    convolution5 = layers.conv2d(inputs=pool3,
                                    num_outputs=512,
                                    kernel_size=[3, 3],
                                    padding='SAME',
                                    activation_fn=tf.nn.relu)
    #max pooling 4 512*2*55 -> 512*1*55
    #pool4 = layers.max_pool2d(inputs=convolution5, kernel_size=[2, 1], stride=[2, 1])

    n1 = layers.batch_norm(convolution5)

    convolution6 = layers.conv2d(inputs=n1,
                                     num_outputs=512,
                                     kernel_size=[3, 3],
                                     padding='SAME',
                                     activation_fn=tf.nn.relu)
    n2 = layers.batch_norm(convolution6)
    pool4 = layers.max_pool2d(inputs=n2, kernel_size=[2, 1], stride=[2, 1])

    convolution7 = layers.conv2d(inputs=pool4,
                                     num_outputs=512,
                                     kernel_size=[2, 2],
                                     padding='VALID',
                                     activation_fn=tf.nn.relu)


    pool5 = layers.max_pool2d(inputs=convolution7, kernel_size=[2, 1], stride=[2, 1])

    cnn_out = tf.squeeze(pool5, axis=1)
    #cnn_out = tf.reshape(n1,[batch_size, -1, 512])
    #cnn_out = tf.layers.flatten(n1)

    cell = tf.contrib.rnn.GRUCell(num_units=1024)
    enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                cell_bw=cell,
                                                                inputs=cnn_out,
                                                                dtype=tf.float32)
    encoder_outputs = tf.concat(enc_outputs, -1)

    #attention loss
    output_embed = layers.embed_sequence(att_train_output,
                                            vocab_size=VOCAB_ATT_SIZE,
                                            embed_dim=ATT_EMBED_DIM, scope='embed')
    embeddings = tf.Variable(tf.truncated_normal(shape=[VOCAB_ATT_SIZE, ATT_EMBED_DIM],
                                                    stddev=0.1), name='decoder_embedding')
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)

    train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, att_train_length)
    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                            start_tokens=tf.to_int32(start_tokens),
                                                            end_token=1)

    train_outputs = att_decode(train_helper, encoder_outputs, 'decode')
    pred_outputs = att_decode(pred_helper, encoder_outputs, 'decode', reuse=True)


    mask = tf.cast(tf.sequence_mask(batch_size * [att_train_length[0]-1], att_train_length[0]), tf.float32)
    att_loss = tf.contrib.seq2seq.sequence_loss(train_outputs[0].rnn_output, att_target_output,
                                                weights=mask)

    attention_loss = tf.reduce_mean(att_loss)

    #CTC loss
    #project_output = layers.fully_connected(inputs=encoder_outputs,
    #                                            num_outputs=VOCAB_CTC_SIZE + 1,
    #                                            activation_fn=None)

    #ctc_loss = tf.nn.ctc_loss(labels=ctc_label,
    #                              inputs=project_output,
    #                              sequence_length=ctc_feature_length,
    #                              time_major=False)
    #ctc_loss = tf.reduce_mean(ctc_loss)

    global_step = tf.Variable(0, trainable = False)
    #merge part
    t_loss = attention_loss*att_loss_weights 
    #+ ctc_loss*ctc_loss_weights
    train_step = tf.train.AdadeltaOptimizer().minimize(t_loss,global_step=global_step)

    tf.summary.scalar('attention_loss', attention_loss)
    #tf.summary.scalar('ctc_loss', ctc_loss)
    
    #optimizer
    #optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,
    #                                  beta1 = beta1,
    #                                  beta2 = beta2).minimize(loss,
    #                                                          global_step = global_step)
    
    
    #error rate
    #pred_labels = sparse_tuple_from(pred_outputs)

    #acc_rate = tf.reduce_mean(tf.edit_distance(pred_labels, labels, normalize=True))

    #predict_accuracy = 1-acc_rate

    #tf.summary.scalar('predict accuracy', predict_accuracy)

    merged_summary = tf.summary.merge_all()
    

#training
with tf.device('/cpu:0'):
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    with tf.Session(config = config,graph=graph_1) as sess:
        #initialize
        #run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('./logs/chi_5_ctcAtt', sess.graph)
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
                batch_inputs, batch_seq_len, batch_labels, batch_label_len, batch_flatten_labels, batch_att_target_output\
                , batch_att_train_output, batch_att_train_length = \
                        input_data_loader.get_batch_by_index(indexs)

                feed = {inputs: batch_inputs,
                        labels: batch_labels,
                        seq_len: batch_seq_len,
                        label_len: batch_label_len,
                        flatten_labels: batch_flatten_labels,
                        #attention + ctc
                        #ctc_label: batch_labels,
                        #ctc_feature_length: np.ones(batch_size)*512,  #need fixed
                        att_target_output: batch_att_target_output,
                        att_train_length: batch_att_train_length,
                        att_train_output: batch_att_train_output}
                #print(batch_inputs)
                #print(batch_labels)
                #print(batch_seq_len)
                    # if summary is needed
                    # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)

                summary_str, batch_cost, _, step= \
                        sess.run([merged_summary, t_loss, train_step,  global_step], feed) 
#options=run_options, run_metadata=run_metadata)
                    # calculate the cost
                train_cost += batch_cost * batch_size
                if (cur_batch + 1) % 50 == 0:
               	    print('batch', cur_batch, ': time', time.time() - batch_time,'      batch_cost: ',batch_cost)
                summary_writer.add_summary(summary_str, step)

                #tl = timeline.Timeline(run_metadata.step_stats)
                #ctf = tl.generate_chrome_trace_format()
                #with open('timelineGreedy.json', 'w') as f:
                #    f.write(ctf)

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
                    saver.save(sess, "./model/model_10_ctcAtt/model_10.ckpt")
                    acc_batch_total = 0
                    ler = 0
                    lr = 0
                    for j in range(num_batches_per_epoch_val):
                        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                        range(j * batch_size, (j + 1) * batch_size)]
                        val_inputs, val_seq_len, val_labels, val_label_len, val_flatten_labels, val_att_target_output\
                , val_att_train_output, val_att_train_length= \
                                valid_input_data_loader.get_batch_by_index(indexs_val)
                        val_feed = {inputs: val_inputs,
                                    labels: val_labels,
                                    seq_len: val_seq_len,
                                    label_len: val_label_len,
                                    flatten_labels: val_flatten_labels,
                                    #ctc_label: val_labels,
                                    #ctc_feature_length: np.ones(batch_size)*512,  #need fixed
                                    att_target_output: val_att_target_output,
                                    att_train_length: val_att_train_length,
                                    att_train_output: val_att_train_output}

                        valid_loss= \
                                sess.run([t_loss],val_feed)

                        #aver_label_length = val_labels[0].shape[0]/val_labels[2][0]

                        #print(int(batch_labels[0].shape[0]),'  and   ',int(batch_labels[2].shape[0]))

                        #print('aver= ',aver_label_length)

                        #acc_rate = 1-aed/aver_label_length
                            # print the decode result
                        #ri_labels = valid_input_data_loader.the_label(indexs_val)
                        #acc = utils.accuracy_calculation(ori_labels, dense_decoded,
                                                            #ignore_value=-1, isPrint=True)
                        acc_batch_total = acc_batch_total+valid_loss

                    avg_valid_loss = acc_batch_total / num_batches_per_epoch_val

                    avg_train_cost = train_cost / ((cur_batch + 1) * batch_size)

                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                        "valid_loss = {:.3f},avg_train_cost = {:.3f}, " \
                        "time = {:.3f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                    cur_epoch + 1, epochs, avg_valid_loss, avg_train_cost,
                                    time.time()-start_time))

        
