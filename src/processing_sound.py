# processes sound data to spectrogram

from scipy import signal
from scipy.io import wavfile
from os import walk, listdir
import numpy as np

def read(set):

    # paths
    data = './data/'
    path = data + 'vivos/' + set
    path_sound = path + '/waves/'

    for directory in walk(path_sound):
        for person in directory[1]:
            print('processing participant ' + person)
            path_person = path_sound + person + '/'

            for file in listdir(path_person):

                # reads sound file
                rate, sample = wavfile.read(path_person + file)
                _, _, spectrogram = signal.spectrogram(sample, rate)
    
                # writes to file
                np.savetxt(data + 'spectrogram/' + set + '/' + file + '.txt', np.array(spectrogram))

if __name__ == '__main__':
    read('train')
    read('test')
