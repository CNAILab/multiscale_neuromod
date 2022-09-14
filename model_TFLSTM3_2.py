from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops

class LSTMCellNM(Layer):
    def __init__(self,
                 units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(LSTMCellNM, self).__init__(**kwargs)
        self.w = tf.Variable(.02 * numpy.random.rand(units, units) - .01, dtype=tf.float32)
        self.alpha = tf.Variable(.0001 * numpy.random.rand(1, 1, units), dtype=tf.float32)
        #self.hebb = tf.Variable(.0001 * numpy.random.rand(10, units, units), dtype=tf.float32)

        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = [self.units, self.units, (self.units, self.units)]
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent_kernel_h2mod = self.add_weight(
            shape=(self.units, 1),
            name='recurrent_kernel_h2mod',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.recurrent_kernel_fanout = self.add_weight(
            shape=(1, self.units),
            name='recurrent_kernel_h2mod',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    #def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        '''
        shape x_i, x_f, x_c, x_o: (10, 128) (10, 128) (10, 128) (10, 128) = [6, 1149 - batch size, hidden size]
        shape_h_tm1_i (10, 128) - hidden[0]
        shape_i, f, c, o (10, 128) (10, 128) (10, 128) (10, 128) - fgt, ipt, opt
        shape_self.recurrent_kernel[:, :self.units] (128, 128) - linear layer
        shape K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]) (10, 128) - self.h2f(hidden[0])
    
        shape h_tm1_i.unsqueeze(1) (10, 1, 128)
        shape self.w (128, 128)
        shape self.alpha (1, 1, 128)
    
        shape self.w + torch.mul(self.alpha, hebb) (10, 128, 128)
        shape h2coutput (10, 128)
        shape x2coutput_shape (10, 128)
        shape inputstocell_shape (10, 128)
        shape C_shape (10, 128)
        shape C and H (10, 128) (10, 128)
        shape tf.expand_dims(h_tm1_c, 2) (10, 128, 1) 
        shape tf.expand_dims(inputstocell, 1) (10, 1, 128) 
        shape deltahebb (10, 128, 128)
        shape F.tanh(self.h2mod(hactiv)) (10, 1) 
        shape myeta (10, 1, 1)
        
        myeta (10, 1, 128)
        hebb (10, 128, 128)


        '''
    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                array_ops.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        hebb = states[2]
        # print('STATES Are', states)

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        x_i = K.dot(inputs_i, self.kernel[:, :self.units])
        x_f = K.dot(inputs_f, self.kernel[:, self.units:self.units * 2])
        x_c = K.dot(inputs_c, self.kernel[:, self.units * 2:self.units * 3])
        x_o = K.dot(inputs_o, self.kernel[:, self.units * 3:])

        if self.use_bias:
            x_i = K.bias_add(x_i, self.bias[:self.units])
            x_f = K.bias_add(x_f, self.bias[self.units:self.units * 2])
            x_c = K.bias_add(x_c, self.bias[self.units * 2:self.units * 3])
            x_o = K.bias_add(x_o, self.bias[self.units * 3:])

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(
            x_f + K.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        # c = f * c_tm1 + i * self.activation(
        #    x_c + K.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))

        #c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)

        h2coutput = tf.squeeze(tf.matmul(tf.expand_dims(h_tm1_o, 1), (self.w + self.alpha * hebb)), 1)
        x2coutput = x_c
        inputstocell = self.activation(x_c + h2coutput)

        c = f * c_tm1 + i * inputstocell

        h = o * self.activation(c)

        deltahebb = tf.matmul(tf.expand_dims(h_tm1_c, 2), tf.expand_dims(inputstocell, 1))
        myeta = tf.expand_dims(self.activation(K.dot(h, self.recurrent_kernel_h2mod)), 2)

        #deltahebb = torch.bmm(hidden[0].unsqueeze(2), inputstocell.unsqueeze(1))
        #myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1

        myeta = tf.expand_dims(tf.squeeze(K.dot(myeta, self.recurrent_kernel_fanout)), 1)

        print('Updating hebb')
        hebb = tf.clip_by_value(hebb + myeta * deltahebb, -2.0, 2.0)

        #myeta = self.modfanout(myeta).squeeze().unsqueeze(1)

        #hebb = torch.clamp(hebb + myeta * deltahebb, min=-2.0, max=2.0)

        print('Using NM version of TFLSTMver3_NM!')
        return h, [h, c, hebb]

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'unit_forget_bias':
                self.unit_forget_bias,
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation
        }
        base_config = super(LSTMCellNM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    #def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
     #   return list(_generate_zero_filled_state_for_cell(
      #      self, inputs, batch_size, dtype))

def displaced_linear_initializer(input_size, displace):
    stddev = 1. / numpy.sqrt(input_size)
    return tf.keras.initializers.TruncatedNormal(mean=displace * stddev, stddev=stddev)


class MinimalRNNCell(layers.Layer):
    def __init__(self,
                 target_ensembles,
                 nh_lstm,
                 nh_bottleneck,
                 nh_embed=None,
                 dropoutrates_bottleneck=None,
                 bottleneck_weight_decay=0.0,
                 bottleneck_has_bias=False,
                 init_weight_disp=0.0, **kwargs):

        super(MinimalRNNCell, self).__init__(**kwargs)
        self._target_ensembles = target_ensembles
        self._nh_embed = nh_embed
        self._nh_lstm = nh_lstm
        self._nh_bottleneck = nh_bottleneck
        self._dropoutrates_bottleneck = dropoutrates_bottleneck
        self._bottleneck_weight_decay = bottleneck_weight_decay
        self._bottleneck_has_bias = bottleneck_has_bias
        self._init_weight_disp = init_weight_disp
        self.training = True
        self.state_size = (nh_lstm, nh_lstm, (nh_lstm, nh_lstm))

        # defining layers
        #self.lstm = layers.LSTMCell(units=self._nh_lstm, name="trash")
        self.lstm = LSTMCellNM(units=self._nh_lstm)#, name="trash")

        self.bottleneck = layers.Dense(self._nh_bottleneck, name="bottleneck",
                                       use_bias=self._bottleneck_has_bias,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                           self._bottleneck_weight_decay))
        self.output_layers = []
        for ens in self._target_ensembles:
            dense = layers.Dense(units=ens.n_cells, name="pc_logits",
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self._bottleneck_weight_decay),
                                 kernel_initializer=displaced_linear_initializer(self._nh_bottleneck,
                                                                                 self._init_weight_disp))

            self.output_layers.append(dense)

        #self.dropout = layers.Dropout(self._dropoutrates_bottleneck, name="dropout")

    def call(self, inputs, states):
        conc_inputs = tf.concat(inputs, axis=1)
        lstm_inputs = conc_inputs

        # lstm_inputs: shape=(10, 3)
        # states: [(10, 128), (10, 128)] without hebb

        lstm_output = self.lstm(lstm_inputs, states=states)

        next_state = lstm_output[1:][0]
        lstm_output = lstm_output[0]
        bottleneck = self.bottleneck(lstm_output)
        if self.training and self._dropoutrates_bottleneck is not None:
            #bottleneck = self.dropout(bottleneck)
            tf.logging.info("Adding dropout layers to TFLSTM ver3")
            n_scales = len(self._dropoutrates_bottleneck)
            scale_pops = tf.split(bottleneck, n_scales, axis=1)
            dropped_pops = [tf.nn.dropout(pop, rate, name="dropout")
                            for rate, pop in zip(self._dropoutrates_bottleneck,
                                                 scale_pops)]
            bottleneck = tf.concat(dropped_pops, axis=1)
            #print('Added dropout from TFLSTM ver3!')
        ens_outputs = [
            layer(bottleneck)
            for layer in self.output_layers]
        return (ens_outputs, bottleneck, lstm_output), tuple(list(next_state))


class GridCellNetwork(tf.keras.models.Model):
    def __init__(
            self,
            target_ensembles,
            nh_lstm,
            nh_bottleneck,
            dropoutrates_bottleneck,
            bottleneck_weight_decay,
            bottleneck_has_bias,
            init_weight_disp,
            **kwargs):
        super(GridCellNetwork, self).__init__(**kwargs)

        self._target_ensembles = target_ensembles
        self._nh_lstm = nh_lstm
        self._nh_bottleneck = nh_bottleneck
        self._dropoutrates_botleneck = dropoutrates_bottleneck
        self._bottleneck_weight_decay = bottleneck_weight_decay
        self._bottleneck_has_bias = bottleneck_has_bias
        self._init_weight_disp = bottleneck_has_bias

        self.init_lstm_state = layers.Dense(self._nh_lstm, name="state_init")
        self.init_lstm_cell = layers.Dense(self._nh_lstm, name="cell_init")
        self.rnn_core = MinimalRNNCell(
            target_ensembles=target_ensembles,
            nh_lstm=nh_lstm,
            nh_bottleneck=nh_bottleneck,
            dropoutrates_bottleneck=dropoutrates_bottleneck,
            bottleneck_weight_decay=bottleneck_weight_decay,
            bottleneck_has_bias=bottleneck_has_bias,
            init_weight_disp=init_weight_disp
        )
        self.RNN = layers.RNN(return_state=True, return_sequences=True, cell=self.rnn_core)

    def call(self, velocities, initial_conditions, trainable=False):
        concat_initial = tf.concat(initial_conditions[:-1], axis=1)
        init_lstm_state = self.init_lstm_state(concat_initial)
        init_lstm_cell = self.init_lstm_cell(concat_initial)
        output_seq = self.RNN((velocities,), initial_state=(init_lstm_state,
                                                            init_lstm_cell,
                                                            initial_conditions[2]))

        '''
        velocities SHAPE (10, 100, 3)
        initial_conditions SHAPE [(10, 256), (10, 12)]
        concat_initial SHAPE (10, 268)
        init_lstm_state SHAPE (10, 128)
        init_lstm_cell SHAPE (10, 128)
        '''
        #print('velocities SHAPE', velocities,
         #     'initial_conditions SHAPE', initial_conditions,
          #    'concat_initial SHAPE', concat_initial,
           #   'init_lstm_state SHAPE', init_lstm_state,
            #  'init_lstm_cell SHAPE', init_lstm_cell)
        final_state = output_seq[-2:]

        output_seq = output_seq[0]

        ens_targets = output_seq[0]
        bottleneck = output_seq[1]
        lstm_output = output_seq[2]

        print("GridCellNetwork in TFLSTMver_NM being used!")

        return (ens_targets, bottleneck, lstm_output), final_state