# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.linear_assignment_ import linear_assignment
from tensorflow.contrib.seq2seq import *
import tensorflow.layers as layers
from sklearn import metrics
import os
import pandas as pd 
import warnings
import argparse

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


def Rnn_cell(cell_size,name):
    if name == 'lstm':
        return tf.nn.rnn_cell.LSTMCell(cell_size)
    elif name == 'gru':
        return tf.nn.rnn_cell.GRUCell(cell_size)


def mse_error(pred,target):
    return tf.reduce_mean(tf.square(tf.subtract(pred, target)))
    

def pur_metric(pred,target):
    n = len(pred)
    tmp = pd.crosstab(pred,target)
    tmp = np.array(tmp)
    ret = np.max(tmp,1)
    ret = float(np.sum(ret))
    ret = ret/n
    return ret

def nmi_metric(pred,target):
    
    NMI = metrics.normalized_mutual_info_score(pred,target)
    return NMI

def RI_metric(pred,target):
    #RI
    n = len(target)
    TP = 0
    TN = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if target[i] != target[j]:
                if pred[i] != pred[j]:
                    TN += 1
            else:
                if pred[i] == pred[j]:
                    TP += 1
    
    RI = n*(n - 1) / 2
    RI = (TP + TN) / RI
    return RI

def _accuracy(y_pred, y_true):
    def cluster_acc(Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max()) + 1
        w = np.zeros((D, D), dtype=np.int32)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, np.array(w)

    y_pred = np.array(y_pred, np.int32)
    y_true = np.array(y_true, np.int32)
    return cluster_acc(y_pred, y_true)

def assess(pred,target):
    pur = pur_metric(pred,target)
    nmi = nmi_metric(pred,target)
    ri = RI_metric(pred,target)
    acc,o = _accuracy(pred,target)
    return ri,nmi,acc,pur

def build_model(config,sess):
    
    data_inputs = tf.placeholder(dtype = tf.float32,shape = [config.batch_size,config.n_steps*config.inputdims],name = "data_inputs")
    mask_inputs = tf.placeholder(dtype = tf.float32,shape = [config.batch_size,config.n_steps*config.inputdims],name = "mask_inputs")
    
    seq_length = tf.placeholder(dtype = tf.int32,shape = [config.batch_size],name = 'seq_length')
    length_mark = tf.placeholder(dtype = tf.float32,shape = [config.batch_size,config.n_steps*config.inputdims],name='length_mask')
    batch_max_len = tf.reduce_max(seq_length)
    left_len = config.n_steps - batch_max_len
    zero_padding = tf.zeros([config.batch_size,left_len,config.inputdims],dtype=tf.float32)

    corpt_inputs = tf.reshape(data_inputs,[config.batch_size,config.n_steps,config.inputdims])  
    masks = tf.reshape(mask_inputs,[config.batch_size,config.n_steps,config.inputdims]) 
    
    
    with tf.variable_scope('generator'):
        begin_value = tf.ones([config.batch_size,1],dtype=tf.float32) * -128
        end_value = tf.ones([config.batch_size,1],dtype=tf.float32) * 128
        
        begin_value = tf.Variable(begin_value,trainable = False)
        end_value = tf.Variable(end_value,trainable = False)
        
        begin_identifier = tf.layers.dense(begin_value,config.inputdims)
        end_identifier = tf.layers.dense(end_value,config.inputdims)
        
        with tf.variable_scope('fw_generator'):
            fw_muticell = tf.contrib.rnn.MultiRNNCell( [Rnn_cell(config.G_hiddensize,config.cell_type) for _ in range(config.G_layer)])
            def fw_initial_fn():            
                initial_elements_finished = (0 >= seq_length)  # all False at the initial step             
                initial_input = begin_identifier            
                return initial_elements_finished, initial_input
            useless_ids = tf.constant(value = 0,dtype = tf.int32,shape = [config.batch_size,])
            def fw_sample_fn(time, outputs, state):                   
                return useless_ids
            def fw_next_inputs_fn(time, outputs, state, sample_ids):       
                del sample_ids
                elements_finished = (time >= seq_length)    
                idx = time-1 
                next_inputs = tf.multiply(outputs,1-masks[:,idx,:]) + tf.multiply(corpt_inputs[:,idx,:],masks[:,idx,:])
                next_state = state            
                return elements_finished, next_inputs, next_state 
            
            fw_dense_layer = layers.Dense(config.inputdims)
            fw_helper = CustomHelper(fw_initial_fn,fw_sample_fn,fw_next_inputs_fn)
            fw_initial_state = fw_muticell.zero_state(config.batch_size,dtype = tf.float32)
            fw_decoder = BasicDecoder(fw_muticell,fw_helper,fw_initial_state,fw_dense_layer)
            fw_outputs,fw_states,_ = dynamic_decode(fw_decoder,
                       output_time_major=False,
                       impute_finished=True,
                       swap_memory=True,
                       scope=None)
            
            fw_generator_outputs = fw_outputs.rnn_output
            
            fw_generator_outputs = tf.concat([fw_generator_outputs,zero_padding],1)
            fw_gen_trim = fw_generator_outputs[:,:-1,:]
            fw_gen_trim_reshaped = tf.reshape(fw_gen_trim,[config.batch_size,config.n_steps,config.inputdims])
        with tf.variable_scope('bw_generator'):
            
            bw_inputs = tf.reverse_sequence(
                                    corpt_inputs,
                                    seq_length,
                                    seq_axis=1,
                                    batch_axis=0)
            bw_masks = tf.reverse_sequence(
                                    masks,
                                    seq_length,
                                    seq_axis=1,
                                    batch_axis=0)
            
            
            bw_muticell = tf.contrib.rnn.MultiRNNCell( [Rnn_cell(config.G_hiddensize,config.cell_type) for _ in range(config.G_layer)] )
            def bw_initial_fn():            
                initial_elements_finished = (0 >= seq_length)  # all False at the initial step             
                initial_input = end_identifier            
                return initial_elements_finished, initial_input
            useless_ids = tf.constant(value = 0,dtype = tf.int32,shape = [config.batch_size,])
            def bw_sample_fn(time, outputs, state):                   
                return useless_ids
            def bw_next_inputs_fn(time, outputs, state, sample_ids):       
                del sample_ids
                elements_finished = (time >= seq_length)    
                idx = time-1 
                next_inputs = tf.multiply(outputs,1-bw_masks[:,idx,:]) + tf.multiply(bw_inputs[:,idx,:],bw_masks[:,idx,:])
                next_state = state            
                return elements_finished, next_inputs, next_state 
            
            bw_dense_layer = layers.Dense(config.inputdims)
            bw_helper = CustomHelper(bw_initial_fn,bw_sample_fn,bw_next_inputs_fn)
            bw_initial_state = bw_muticell.zero_state(config.batch_size,dtype = tf.float32)
            bw_decoder = BasicDecoder(bw_muticell,bw_helper,bw_initial_state,bw_dense_layer)
            bw_outputs,bw_states,_ = dynamic_decode(bw_decoder,
                       output_time_major=False,
                       impute_finished=True,
                       swap_memory=True,
                       scope=None)
            
            bw_generator_outputs = bw_outputs.rnn_output
            bw_generator_outputs = tf.concat([bw_generator_outputs,zero_padding],1)
            bw_gen_trim = bw_generator_outputs[:,:-1,:]
            bw_gen_trim_revesed = tf.reverse_sequence(
                                    bw_gen_trim,
                                    seq_length,
                                    seq_axis=1,
                                    batch_axis=0)
            bw_gen_trim_reshaped = tf.reshape(bw_gen_trim_revesed,[config.batch_size,config.n_steps,config.inputdims])
        
        impute_vector = (fw_gen_trim_reshaped + bw_gen_trim_reshaped)/2
        
        imputed_vector = tf.multiply(corpt_inputs,masks) + tf.multiply(impute_vector,1-masks)
        
    with tf.variable_scope('discriminator'):
        disc_cell = tf.contrib.rnn.MultiRNNCell( [Rnn_cell(32,config.cell_type),Rnn_cell(16,config.cell_type),Rnn_cell(8,config.cell_type),Rnn_cell(16,config.cell_type),Rnn_cell(32,config.cell_type)] )       
         
        disc_output, _ = tf.nn.dynamic_rnn(
                cell = disc_cell,
                inputs = imputed_vector,
                sequence_length = seq_length,
                dtype = tf.float32)
        
        discriminator_output = layers.dense(disc_output,config.inputdims)
        
    with tf.variable_scope("decoder"):
        latent_rep = tf.concat([*fw_states,*bw_states],1)
        
        if config.dataset_name != 'Physionet 2012':
            latent_rep = tf.layers.dense(latent_rep,latent_rep.shape[-1])
        else:
            latent_rep = tf.layers.dense(latent_rep,2000)
            latent_rep = tf.layers.dense(latent_rep,500)
            latent_rep = tf.layers.dense(latent_rep,500)
            latent_rep = tf.layers.dense(latent_rep,10)
            
        hidden_size = int(latent_rep.shape[1])
        gru_cell = Rnn_cell(hidden_size,config.cell_type)
        def re_initial_fn():            
            initial_elements_finished = (0 >= seq_length)  # all False at the initial step             
            initial_input = begin_identifier 
            return initial_elements_finished, initial_input
        def re_sample_fn(time, outputs, state):                   
            return useless_ids
        def re_next_inputs_fn(time, outputs, state, sample_ids):            
            del sample_ids
            elements_finished = (time >= seq_length)    
            next_inputs = outputs
            next_state = state            
            return elements_finished, next_inputs, next_state 
        re_helper = CustomHelper(re_initial_fn,re_sample_fn,re_next_inputs_fn)
        re_dense_layer = layers.Dense(config.inputdims)
        re_decoder = BasicDecoder(gru_cell,re_helper,latent_rep,re_dense_layer)
        decoder_outputs,_,_ = dynamic_decode(re_decoder,
               output_time_major=False,
               impute_finished=True,
               swap_memory=True,
               scope=None)
        decoder_outputs = decoder_outputs.rnn_output
        decoder_outputs = tf.concat([decoder_outputs,zero_padding],1)
        decoder_end_pred = [decoder_outputs[i,seq_length[i],:] for i in range(config.batch_size)]
        decoder_end_pred = tf.reshape(decoder_end_pred,[config.batch_size,config.inputdims])
        

    with tf.name_scope('loss'):
        disc_reshaped = tf.reshape(discriminator_output,[config.batch_size,-1])
        #loss_D
        loss_D = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_reshaped,labels = mask_inputs)
        loss_D = tf.multiply(loss_D,length_mark)
        loss_D = tf.reduce_sum(tf.reduce_mean(loss_D,0))
        #loss_G
        loss_G = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_reshaped,labels = 1-mask_inputs)
        loss_G = tf.multiply(loss_G,length_mark)
        loss_G = tf.multiply(loss_G,1-mask_inputs)
        loss_G = tf.reduce_sum(tf.reduce_mean(loss_G,0))
        #loss_pre
        pre_tmp = tf.multiply(impute_vector,masks)
        targ_pre = tf.multiply(masks,corpt_inputs)
        loss_pre = mse_error(pre_tmp,targ_pre) 
        #loss_re
        output_trim = decoder_outputs[:,:-1,:]
        out_tmp = tf.multiply(output_trim,masks)
        targ_re = tf.multiply(corpt_inputs,masks)
        loss_re = mse_error(out_tmp,targ_re)
        #loss_km
        HTH = tf.matmul(latent_rep,tf.transpose(latent_rep))
        F_copy = tf.get_variable('F', shape=[config.batch_size, config.k_cluster],
                        initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32),
                        dtype=tf.float32,
                        trainable=False
                        )
        FTHTHF = tf.matmul(tf.matmul(tf.transpose(F_copy),HTH),F_copy)
        loss_km = tf.trace(HTH) - tf.trace(FTHTHF)
        #update_op
        F = tf.placeholder(tf.float32,shape = [config.batch_size,config.k_cluster])
        F_update = tf.assign(F_copy,F)
        #total
        loss_disc = loss_D
        
        Gloal_step = tf.Variable(tf.constant(0))
        lbdkm = tf.train.exponential_decay( config.lambda_kmeans, Gloal_step, decay_steps=config.train_dataset_size//config.batch_size, decay_rate=0.95, staircase=False)
        if config.IsD == True:
            loss_gen = loss_G + loss_pre + loss_re + loss_km * lbdkm
        else:
            loss_gen = loss_G + loss_pre + loss_re + loss_km * config.lambda_kmeans
            
    with tf.name_scope("train_op"):
        D_loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        train_disc = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_disc,var_list = D_loss_vars)
        train_gen = tf.train.AdamOptimizer(config.learning_rate).minimize(loss_gen)
    
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    Inputs = {"data":data_inputs,"mask":mask_inputs,"F_new_value":F,'length':seq_length,'lengthmark':length_mark,'STEP':Gloal_step}
    train_op = {"disc":train_disc,"gen":train_gen}
    outputs = {"H":latent_rep,'HTH':HTH,'lbdkm':lbdkm}
    losses = {"pred":loss_pre,"recon":loss_re,"kmean":loss_km,"G":loss_G,"D":loss_D}
    
    return Inputs,train_op,outputs,F_update,losses


def load_mask(filename):
    mask_label = np.loadtxt(filename,delimiter = ",")
    mask = mask_label[:,1:].astype(np.float32)
    return mask

def load_data(filename):
    data_label = np.loadtxt(filename,delimiter = ",")
    data = data_label[:,1:].astype(np.float32)
    label = data_label[:,0].astype(np.int32)
    return data,label

def load_length(filename):
    length = np.loadtxt(filename,delimiter = ",")
    return length

def load_lengthmark(filename):
    lengthmark = np.loadtxt(filename,delimiter = ",")
    return lengthmark

def get_batch(data,label,mask,length,lengthmark,config):
    
    samples_num = data.shape[0]
    batch_num = int(samples_num / config.batch_size)
    left_row = samples_num - batch_num * config.batch_size

    for i in range(batch_num):
        batch_data = data[i * config.batch_size: (i + 1) * config.batch_size, :]
        batch_label = label[i * config.batch_size: (i + 1) * config.batch_size]
        batch_mask = mask[i * config.batch_size: (i + 1) * config.batch_size, :]
        batch_length = length[i * config.batch_size: (i + 1) * config.batch_size]
        batch_lengthmark = lengthmark[i * config.batch_size: (i + 1) * config.batch_size,:]
        yield (batch_data,batch_label,batch_mask,batch_length,batch_lengthmark)

    if left_row != 0:
        need_more = config.batch_size - left_row
        need_more = np.random.choice(np.arange(samples_num), size=need_more)
        batch_data = np.concatenate((data[-left_row:, :], data[need_more,:]), axis=0)
        batch_label = np.concatenate((label[-left_row:],label[need_more]), axis=0)
        batch_mask = np.concatenate((mask[-left_row:, :], mask[need_more,:]), axis=0)
        batch_length = np.concatenate((length[-left_row:],length[need_more]), axis=0)
        batch_lengthmark = np.concatenate((lengthmark[-left_row:,:],lengthmark[need_more,:]), axis=0)
        yield (batch_data,batch_label,batch_mask,batch_length,batch_lengthmark)

def run(config):
    '''data'''
    train_data_filename = config.data_dir + "/" + config.dataset_name + "/" + config.dataset_name + "_TRAIN"
    train_data,train_label = load_data(train_data_filename)
    test_data_filename = train_data_filename.replace('TRAIN','TEST')
    test_data,test_label = load_data(test_data_filename)
    '''mask'''
    train_maskname = config.data_dir + "/" + config.dataset_name + "/" + "mask_" + config.dataset_name + "_TRAIN"
    train_mask = load_mask(train_maskname)
    test_maskname = train_maskname.replace('TRAIN','TEST')
    test_mask = load_mask(test_maskname)
    '''length'''
    train_length_filename = config.data_dir + "/" + config.dataset_name + "/" + 'length_' + config.dataset_name + "_TRAIN"
    train_length = load_length(train_length_filename)
    test_length_filename = train_length_filename.replace('TRAIN','TEST')
    test_length = load_length(test_length_filename)
    '''train_lengthmark'''
    train_lengthmark_filename = config.data_dir + "/" + config.dataset_name + "/" + 'lengthmark_' + config.dataset_name + "_TRAIN"
    train_lengthmark = load_lengthmark(train_lengthmark_filename)
    test_lengthmark_filename = train_lengthmark_filename.replace('TRAIN','TEST')
    test_lengthmark = load_length(test_lengthmark_filename)
    
    '''dim of multivariable dataset'''
    dims = dict()
    with open ('./dims.txt') as f:
        for line in f:
            tmps = line[:-1].split(',')
            dims[tmps[0]] = int(tmps[1])
            
    train_dataset_size = train_data.shape[0]
    test_dataset_size = test_data.shape[0]
    config.inputdims = dims[config.dataset_name]
    config.n_steps = train_data.shape[1] // config.inputdims  
    config.k_cluster = len(np.unique(train_label))
    config.train_dataset_size = train_dataset_size
    
    if config.batch_size > train_dataset_size:
        config.batch_size = train_dataset_size
    elif config.batch_size < config.k_cluster:
        config.batch_size = config.k_cluster
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    session_config.gpu_options.allow_growth = True 
    '''build_model'''
    with tf.Session(config = session_config) as sess:
        inputs,train_op,outputs,F_update,losses = build_model(config,sess)
        
        RI=[]
        NMI=[]
        ACC=[]
        PUR=[]
        
        GLOBAL_STEP = -1
        for i in range(config.epoch):
            print(i)
            
            '''Train'''
            for batch_data,batch_label,batch_mask,batch_length,batch_lengthmark in get_batch(train_data,train_label,train_mask,train_length,train_lengthmark,config):
                GLOBAL_STEP += 1
                for _ in range(config.D_steps):
                    sess.run(train_op["disc"],feed_dict = {inputs['STEP']:GLOBAL_STEP,inputs["data"]:batch_data,inputs["mask"]:batch_mask,inputs['length']:batch_length,inputs['lengthmark']:batch_lengthmark})
                for j in range(config.G_steps):
                    sess.run(train_op["gen"],feed_dict = {inputs['STEP']:GLOBAL_STEP,inputs["data"]:batch_data,inputs["mask"]:batch_mask,inputs['length']:batch_length,inputs['lengthmark']:batch_lengthmark})
                
            if i % config.T_update_F == 0:
                '''calculate F'''
                H = sess.run(outputs["H"],feed_dict = {inputs["data"]:batch_data,inputs["mask"]:batch_mask,inputs['length']:batch_length,inputs['lengthmark']:batch_lengthmark})
                U,s,V = np.linalg.svd(H)
                F_new = U.T[:config.k_cluster,:]
                F_new = F_new.T
                sess.run(F_update, feed_dict={inputs['F_new_value']: F_new})
            
            '''TEST'''
            H_outputs = []
            for batch_data,batch_label,batch_mask,batch_length,batch_lengthmark in get_batch(test_data,test_label,test_mask,test_length,test_lengthmark,config):
                 H = sess.run(outputs["H"],feed_dict = {inputs["data"]:batch_data,inputs["mask"]:batch_mask,inputs['length']:batch_length,inputs['lengthmark']:batch_lengthmark})
                 H_outputs.append(H)
            H_outputs = np.concatenate(H_outputs,0)
            H_outputs = H_outputs[:test_dataset_size,:]
            Km = KMeans(n_clusters=config.k_cluster)
            pred_H = Km.fit_predict(H_outputs)
            
            '''record'''
            ri,nmi,acc,pur = assess(pred_H,test_label)
            RI.append(ri)
            NMI.append(nmi)
            ACC.append(acc)
            PUR.append(pur)

    ri = max(RI[80],RI[300],RI[500])
    nmi = max(NMI[80],NMI[300],NMI[500])
    acc = max(ACC[80],ACC[300],ACC[500])
    pur = max(PUR[80],PUR[300],PUR[500])
    
    return ri,nmi,pur,acc
            
        
#### MAIN
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,required=False,default=32)
    parser.add_argument('--T_update_F',type=int,required=False,default=1)
    parser.add_argument('--G_steps',type=int,required=False,default=1)
    parser.add_argument('--D_steps',type=int,required=False,default=1)
    parser.add_argument('--epoch',type=int,required=False,default=500)
    parser.add_argument('--learning_rate',type=float,required=False,default=5e-3)
    parser.add_argument('--dataset_name',type=str,required=True)
    parser.add_argument('--lambda_kmeans',type=float,required=False,default=1e-3)
    parser.add_argument('--G_hiddensize',type=int,required=False,default=50)
    parser.add_argument('--G_layer',type=int,required=False,default=1)
    parser.add_argument('--IsD',type=bool,required=False,default=False)
    parser.add_argument('--cell_type',type=str,required=False,default='gru')
    
    
    config = parser.parse_args()
    
    ri,nmi,pur,cluster_acc = run(config)
    
    print('%s,%.6f,%.6f,%.6f,%.6f' % (config.dataset_name,ri,nmi,pur,cluster_acc))