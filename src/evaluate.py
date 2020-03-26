# evaluates the accuracy of the model

from tensorflow.keras.models import model_from_json, Model
from model import clipped_ReLU, loss, build, process
from os import listdir
from numpy import loadtxt, array, argmax
from processing_text import decode
import tensorflow as tf
from tensorflow import transpose

# data path
path = './data/'

# load model
def load(p):
    model = build()
    model.load_weights(p + 'model/weight.h5')
    return model

# converts softmax output to text
def out(list):
    output = []
    for i in range(0, len(list)):
        if list[i] != '_':
            if i > 0:
                if list[i] != list[i - 1]:
                    output.append(list[i])
            else:
                output.append(list[i])
    return output

# processes sound spectrograms
def process_sound(sound, model):
    sound = sound.T
    length_sound = sound.shape[0]
    sound = sound.reshape((1, length_sound, 129))
    
    # predicts using another model to get the softmax layer output
    # f = function([model.get_layer('input').input], [model.get_layer('ctc').input[0]])
    # a = f([sound])
    model2 = Model(inputs = model.get_layer('input').input, outputs = model.get_layer('ctc').input[0])
    prediction = model2.predict(sound)

    # gets highest probability from softmax layer
    # text_predict = []
    # for i in range(prediction.shape[1]):
    #     text_predict.append(argmax(prediction[0][i]))

    # beam search
    prediction = transpose(prediction, perm = [1, 0, 2])    # tf.nn.ctc_beam_search_decoder expects size
                                                            # [max_time, batch_size, num_classes]
    logits = tf.math.log(prediction)
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, [prediction.shape[0]])
    decoded = tf.sparse.to_dense(decoded[0])
        
    return out(decode(decoded[0]))

# evaluates model on test data
def evaluate(model):

    # loads test data
    file_txt = open(path + 'encoded/train.txt')
    path_sound = path + 'spectrogram/train/'
    for file_sound in listdir(path_sound):

        # sound spectrogram
        sound = loadtxt(path_sound + '/' + file_sound)

        # prompt
        txt = file_txt.readline().strip('\n').split(' ')
        txt = [int(i) for i in txt[:-1]]
        # txt = array(process(file_txt))

        print(''.join(process_sound(sound, model)) + '\t' + ''.join(decode(txt)))
        # prediction = model.predict([sound, txt, array([sound.shape[0]]), array([txt.shape[0]])])
        # print(prediction)

if __name__ == '__main__':
    evaluate(load(path))

