# machine learning model

from os import listdir
from numpy import loadtxt, pad, array, zeros
from tensorflow.keras.layers import TimeDistributed, Dense, Bidirectional, LSTM, Dropout, Input, Masking, Lambda
from tensorflow.keras.backend import relu, ctc_batch_cost, function
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from math import ceil

path = './data/'
path_save = path + 'model/'

# model hyperparameters

#   set of dense layers
#       number of units in each layer
set_1 = [128, 128]
#       parameters initializer
ini_param_1 = 'random_normal'
#       bias initializer
ini_bias_1 = 'random_normal'
#       drop out rate
rate_1 = 0.2

#   set of bidirectional LSTM layers
#       number of units in each layer
set_LSTM = [128]
#       activation function
f_LSTM = 'relu'
#       parameters initializer
ini_param_LSTM = 'glorot_uniform'
#       bias initializer
ini_bias_LSTM = 'random_normal'
#       merge mode
mode = 'sum'

#   a dense layer
#       number of units
no = 128
#       parameters initializer
ini_param_2 = 'random_normal'
#       bias initializer
ini_bias_2 = 'random_normal'
#       drop out rate
rate_2 = 0.2
#       activation function
f_2 = 'relu'

#   output layer
#       parameters initializer
ini_param_out = 'random_normal'
#       bias initializer
ini_bias_out = 'random_normal'

#   learning rate reduction
factor = 0.2
#       number of epochs before reduction
no_reduction = 5
#       minimum learning rate
rate_min = 0.0000001

#   early stopping
#       minimum change in loss
change = 0
#       number of epochs before reduction
no_stop = 5

optimizer = Adam(epsilon=1e-8)

#   batch size
size_batch = 16

#   number of epochs
no_epoch = 1

#   starts training from the start
start = False

# processes text data
def process(file):
    line = file.readline().strip('\n').split(' ')
    line = [int(no) for no in line[:-1]] # last element is a blank
    line = [97 if no == -1 else no for no in line] # keras doesn't except negative labels
    return line
    
# data generator
def generator(set, size):
    sound_batch = []
    text_batch = []
    sound_length = []
    text_length = []

    # examples in batch
    count = 0

    # max sound number of timesteps in a batch
    max = 0

    current = 1

    # sound data (processed) path
    path_sound = path + 'spectrogram/' + set

    while True:
        text = open(path + 'encoded/' + set + '.txt')
        for file in listdir(path_sound):
            count += 1

            # loads sound data
            sound = loadtxt(path_sound + '/' + file)
            sound_batch.append(sound)
            timestep = sound.shape[1]
            sound_length.append(timestep)

            # keeps track of max number of timesteps in a batch
            if timestep > max:
                max = timestep

            # loads text data
            line = process(text)
            text_batch.append(line)
            text_length.append(len(line))

            # end of batch
            if count == size:

                # padding
                sound_batch = [pad(s, [(0, 0), (0, max - s.shape[1])], mode='constant').T for s in sound_batch]

                input = {'input': array(sound_batch), 'labels': pad_sequences(text_batch, padding='post'),
                            'length_input': array(sound_length), 'length_labels': array(text_length)}

                # dummy output
                out = {'ctc': zeros([size])}

                print('batch ' + str(current) + '/' + str(ceil(11660 / size)))
                current += 1
                yield input, out

                # resets
                count = 0
                sound_batch = []
                text_batch = []
                sound_length = []
                text_length = []
                max = 0

        text.close()

# clipped ReLU
def clipped_ReLU(value):
    return relu(value, max_value=20)

# CTC loss
def loss(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(labels, y_pred, input_length, label_length)

# builds model
def build():

    # input layer 
    input = Input(shape=(None, 129), name='input')

    # CTC input layer
    labels = Input(shape = [None], name='labels')
    input_length = Input(shape = [1], name='length_input')
    label_length = Input(shape = [1], name='length_labels')

    # masking layer
    neural = Masking()(input)

    # set of dense layers
    for layer in set_1:
        neural = TimeDistributed(Dense(layer, kernel_initializer=ini_param_1, bias_initializer=ini_bias_1,
                                        activation=clipped_ReLU))(neural)
        neural = TimeDistributed(Dropout(rate_1))(neural)

    # set of LSTM layers
    for layer in set_LSTM:
        neural = Bidirectional(LSTM(layer, kernel_initializer=ini_param_LSTM, bias_initializer=ini_bias_LSTM,
                                            unit_forget_bias=True, return_sequences=True), merge_mode=mode)(neural)

    # another dense layer
    neural = TimeDistributed(Dense(no, kernel_initializer=ini_param_2, bias_initializer=ini_bias_2, activation=f_2))(neural)
    neural = TimeDistributed(Dropout(rate_2))(neural)

    # output layer
    out = TimeDistributed(Dense(99, kernel_initializer=ini_param_out, bias_initializer=ini_bias_out,
                                activation='softmax'))(neural)

    # CTC loss layer
    loss_layer = Lambda(loss, (1, ), name = 'ctc')([out, labels, input_length, label_length])

    model = Model(inputs = [input, labels, input_length, label_length], outputs = loss_layer)

    # dummy loss function
    f = {'ctc': lambda labels, prediction: prediction}

    model.compile(loss = f, optimizer=optimizer)

    # loads weight to continue to train
    if not start:
        model.load_weights(path_save + 'weight.h5')

    print(model.summary())
    return model

def train(model):

    # reduces learning rate
    reduction = ReduceLROnPlateau(factor=factor, patience=no_reduction, min_lr=rate_min)

    # early stopping
    stop = EarlyStopping(min_delta=0, patience=no_stop, mode='auto')

    # check point
    point = ModelCheckpoint(path_save + 'best.hdf5', save_best_only=True)

    # trains model
    model.fit_generator(callbacks = [reduction, stop, point], generator = generator('train', size_batch), epochs = no_epoch,
                        verbose = 2, steps_per_epoch = ceil(11660 / size_batch))

    # saves weights
    model.save_weights(path_save + 'weight.h5')

if __name__ == '__main__':
    train(build())