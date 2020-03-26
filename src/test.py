# tests the result of the system on sample file

from scipy import signal
from scipy.io.wavfile import read
from evaluate import process_sound, load

# data path
path = './data/'

def test():
    # rate, sample = read(path + '../../data/vivos/train/waves/VIVOSSPK01/VIVOSSPK01_R001.wav')
    rate, sample = read(path + 'test/1.wav')
    _, _, spectrogram = signal.spectrogram(sample[:,0][0::6], rate)

    # print(sample[:,0].shape)

    print(''.join(process_sound(spectrogram, load(path))))

if __name__ == '__main__':
    test()