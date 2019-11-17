import pandas
import pickle
import torch
import torchaudio

from collections import defaultdict
from glob import glob
from torch.nn import ConstantPad1d
from torchaudio.transforms import MelSpectrogram

dialects = ['DR' + str(i) for i in range(1, 9)]
phoneme_cols = ['start', 'end', 'phoneme']
spec = MelSpectrogram(n_fft=1600, f_max=20000)
padding = ConstantPad1d((400, 400), 0)


def preprocess(pkl_path='timit_tokenized.pkl'):
    phoneme_samples = {
        'TRAIN': {x: defaultdict(list) for x in dialects},
        'TEST': {x: defaultdict(list) for x in dialects}
    }

    with torch.no_grad():
        for dataset in ['TRAIN', 'TEST']:
            for dialect in dialects:
                speakers = glob(f'TIMIT/{dataset}/{dialect}/M*')
                for speaker in speakers:
                    sentences = set(path.split('/')[-1][:-4] for path in glob(speaker + '/*'))
                    for sentence in sentences:
                        current_path = f'TIMIT/{dataset}/{dialect}/{speaker.split("/")[-1]}/{sentence}'
                        sample, _ = torchaudio.load_wav(current_path + '.WAV')
                        df_sample = pandas.read_csv(current_path + '.PHN', sep=' ', names=phoneme_cols)
                        for _, row in df_sample.iterrows():
                            subsample = sample[:, row[0]:row[1]]
                            if subsample.shape[1] <= spec.win_length // 2:
                                subsample = padding(subsample)
                            phoneme_samples[dataset][dialect][row[2]].append(spec(subsample).mean(2))

                for phoneme in phoneme_samples[dataset][dialect]:
                    phoneme_samples[dataset][dialect][phoneme] = \
                        torch.cat(phoneme_samples[dataset][dialect][phoneme]).data.numpy()

    with open(pkl_path, 'wb') as f:
        pickle.dump(phoneme_samples, f)


if __name__ == '__main__':
    preprocess()
