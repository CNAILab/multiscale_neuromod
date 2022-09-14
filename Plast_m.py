from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import matplotlib
import numpy as np
import tensorflow as tf
import glob, os
import pandas as pd
#import Tkinter  
import tensorflow.contrib.eager as tfe

matplotlib.use('Agg')

import dataset_reader_split  
import model_TFLSTM3_2 
import scores  
import utils  

FLAGS = {
    'task_dataset_info':'square_room',
    'task_root':'/home/jiemei/scratch/grid_exp_ambulations/grid_cell_datasets',
    'task_env_size':2.2,
    'task_n_pc':[256],
    'task_pc_scale':[0.01],
    'task_n_hdc':[12],
    'task_hdc_concentration':[20.],
    'task_neurons_seed':8341,
    'task_targets_type':'softmax',
    'task_lstm_init_type':'softmax',
    'task_velocity_inputs':True,
    'task_velocity_noise':[0.0, 0.0, 0.0],
    'model_nh_lstm':128,
    'model_nh_bottleneck':256,
    'model_dropout_rates':[0.5],
    'model_weight_decay':1e-5,
    'model_bottleneck_has_bias':False,
    'model_init_weight_disp':0.0,
    'training_epochs':1020,
    'training_steps_per_epoch':1000,
    'training_minibatch_size':10,
    'training_evaluation_minibatch_size':4000,
    'training_clipping_function':'utils.clip_all_gradients',
    'training_clipping':1e-5,
    'training_optimizer_class':'tf.compat.v1.train.RMSPropOptimizer',
    'training_optimizer_options':'{"learning_rate": 1e-5,"momentum": 0.9}',
    #'saver_results_directory':"/home/jiemei/scratch/grid_exp_ambulations/results_TFLSTM/trial_dp_lr_Dec13",
    'saver_eval_time':2
}

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session

# per epoch
loss_epoch = list()
loss_std_epoch = list()
hebbian_per_epoch_mean = list()
hebbian_per_epoch_std = list()

ckpt = '/home/jiemei/scratch/train_test/NM/checkpoint/'
figs = '/home/jiemei/scratch/train_test/NM/figures/'

def dir_create(name_dir):
    ckpt_dir = ckpt + name_dir
    figs_dir = figs + name_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
        
    return ckpt_dir, figs_dir

ckpt_dir, figs_dir = dir_create('only_hebbian_6')
    
FLAGS['saver_results_directory'] = figs_dir
#file_path = ckpt_dir+'/loss.csv'
per_epoch_path = ckpt_dir+'/loss_hebb.csv'

if os.path.isfile(per_epoch_path) == True: 
    if os.stat(per_epoch_path).st_size != 0:
        loss_hebb = pd.read_csv(per_epoch_path)
        loss_epoch = loss_hebb['Loss'].tolist()
        loss_std_epoch = loss_hebb['Loss_std'].tolist()
        hebbian_per_epoch_mean = loss_hebb['Hebb'].tolist()
        hebbian_per_epoch_std = loss_hebb['Hebb_std'].tolist()

with tf.Graph().as_default():
    data_reader = dataset_reader_split.DataReader(
        FLAGS['task_dataset_info'], root=FLAGS['task_root'], num_threads=4)

    train_traj = data_reader.read(batch_size=FLAGS['training_minibatch_size'])

    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=FLAGS['task_env_size'],
        neurons_seed=FLAGS['task_neurons_seed'],
        targets_type=FLAGS['task_targets_type'],
        lstm_init_type=FLAGS['task_lstm_init_type'],
        n_pc=FLAGS['task_n_pc'],
        pc_scale=FLAGS['task_pc_scale'])

    head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=FLAGS['task_neurons_seed'],
        targets_type=FLAGS['task_targets_type'],
        lstm_init_type=FLAGS['task_lstm_init_type'],
        n_hdc=FLAGS['task_n_hdc'],
        hdc_concentration=FLAGS['task_hdc_concentration'])
    target_ensembles = place_cell_ensembles + head_direction_ensembles
    
    rnn = model_TFLSTM3_2.GridCellNetwork(
        target_ensembles=target_ensembles,
        nh_lstm=FLAGS['model_nh_lstm'],
        nh_bottleneck=FLAGS['model_nh_bottleneck'],
        dropoutrates_bottleneck=np.array(FLAGS['model_dropout_rates']),
        #dropoutrates_bottleneck=np.array(model_dropout),
        bottleneck_weight_decay=FLAGS['model_weight_decay'],
        bottleneck_has_bias=FLAGS['model_bottleneck_has_bias'],
        init_weight_disp=FLAGS['model_init_weight_disp'],
    )

    input_tensors = []
    init_pos, init_hd, ego_vel, target_pos, target_hd = train_traj
    print('sample_shape', ego_vel.get_shape())
    if FLAGS['task_velocity_inputs']:
        vel_noise = tf.distributions.Normal(0.0, 1.0).sample(
            sample_shape=ego_vel.get_shape()) * FLAGS['task_velocity_noise']
        input_tensors = [ego_vel + vel_noise] + input_tensors
    inputs = tf.concat(input_tensors, axis=2)

    initial_conds = utils.encode_initial_conditions(
        init_pos, init_hd, place_cell_ensembles, head_direction_ensembles)

    hebb = tf.Variable(0 * np.random.rand(10, 128, 128), dtype=tf.float32)
    initial_conds.append(tf.identity(hebb))

    ensembles_targets = utils.encode_targets(
        target_pos, target_hd, place_cell_ensembles, head_direction_ensembles)

    outputs, _ = rnn(inputs, initial_conds)
    #print('INPUTS shape, initial_conds shape', inputs, initial_conds)
    ensembles_logits, bottleneck, lstm_output = outputs

    # Training loss
    pc_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ensembles_targets[0], logits=ensembles_logits[0], name='pc_loss')
    hd_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=ensembles_targets[1], logits=ensembles_logits[1], name='hd_loss')
    total_loss = pc_loss + hd_loss
    train_loss = tf.reduce_mean(total_loss, name='train_loss')

    # Optimization
    optimizer_class = eval(FLAGS['training_optimizer_class'])  # pylint: disable=eval-used
    optimizer = optimizer_class(**eval(FLAGS['training_optimizer_options']))  # pylint: disable=eval-used
    grad = optimizer.compute_gradients(train_loss)
    clip_gradient = eval(FLAGS['training_clipping_function'])  # pylint: disable=eval-used
    clipped_grad = [
        clip_gradient(g, var, FLAGS['training_clipping']) for g, var in grad
    ]
    train_op = optimizer.apply_gradients(clipped_grad)

    # Grid scores
    grid_scores = dict()
    grid_scores['btln_60'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_90'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_60_separation'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['btln_90_separation'] = np.zeros((FLAGS['model_nh_bottleneck'],))
    grid_scores['lstm_60'] = np.zeros((FLAGS['model_nh_lstm'],))
    grid_scores['lstm_90'] = np.zeros((FLAGS['model_nh_lstm'],))

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    masks_parameters = zip(starts, ends.tolist())
    latest_epoch_scorer = scores.GridScorer(20, data_reader.get_coord_range(),
                                            masks_parameters)

    # with tf.train.SingularMonitoredSession() as sess:
    saver = tf.train.Saver(max_to_keep=105)
    with tf.train.MonitoredTrainingSession() as sess:
        
        if os.path.isdir(ckpt_dir):
           if len(os.listdir(ckpt_dir)) > 2:
               list_of_files = glob.glob(ckpt_dir + '/*')
               list_of_files = [i for i in list_of_files if i.split('.')[-1] != 'csv']
               list_of_files = [i for i in list_of_files if i.split('/')[-1] != 'checkpoint']
               latest_file = max(list_of_files, key=os.path.getctime)
               epoch_last = int(latest_file.split(ckpt_dir+'/model_')[-1].split('.ckpt')[0])
               print('Starting from epoch '+str(epoch_last))
               saver.restore(sess, ckpt_dir+'/model_'+str(epoch_last)+'.ckpt')
           else:
               epoch_last = -1
        

        #for epoch in range(FLAGS['training_epochs']):
        for epoch in range(epoch_last+1, FLAGS['training_epochs']):
        #for epoch in range(epoch_last+1, epoch_last+1+10):
            loss_acc = list()
            for _ in range(FLAGS['training_steps_per_epoch']):
                res = sess.run({'train_op': train_op, 'total_loss': train_loss, 'hebbian': hebb})
                loss_acc.append(res['total_loss'])

            tf.logging.info('Epoch %i, mean loss %.5f, std loss %.5f', epoch,
                            np.mean(loss_acc), np.std(loss_acc))
            
            loss_epoch.append(np.mean(loss_acc))
            loss_std_epoch.append(np.std(loss_acc))
            hebbian_per_epoch_mean.append(np.mean(res['hebbian']))
            hebbian_per_epoch_std.append(np.std(res['hebbian']))

            # if epoch % FLAGS['saver_eval_time'] == 0:
            if (epoch+1) % 10 == 0 or epoch == 0:
                #print('saving to file')
                loss_hb = pd.DataFrame({'Loss': loss_epoch, 
                                        'Loss_std': loss_std_epoch,
                                        'Hebb': hebbian_per_epoch_mean,
                                        'Hebb_std': hebbian_per_epoch_std,
                                       })
                
                loss_hb.to_csv(per_epoch_path)
                
                saver.save(get_session(sess), ckpt_dir+'/model_'+str(epoch)+'.ckpt')
                res = dict()
                for _ in range(FLAGS['training_evaluation_minibatch_size'] //
                               FLAGS['training_minibatch_size']):
                    mb_res = sess.run({
                        'bottleneck': bottleneck,
                        'lstm': lstm_output,
                        'pos_xy': target_pos
                    })
                    res = utils.concat_dict(res, mb_res)

                #if epoch % 10 == 0:
                if (epoch+1) % 10 == 0 or epoch == 0:
                    print('Saving figure for epoch ' + str(epoch))
                    filename = 'rates_and_sac_epoch' + str(epoch) + '.pdf'
                    grid_scores['btln_60'], grid_scores['btln_90'], grid_scores[
                        'btln_60_separation'], grid_scores[
                        'btln_90_separation'] = utils.get_scores_and_plot(
                        latest_epoch_scorer, res['pos_xy'], res['bottleneck'],
                        FLAGS['saver_results_directory'], filename)
                    
                    print('Saved figure for epoch ' + str(epoch))
                