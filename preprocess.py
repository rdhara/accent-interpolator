import pandas
import pickle
import torch
import torchaudio

from collections import defaultdict
from glob import glob
from torch.nn import ConstantPad1d
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

dialects = ['DR' + str(i) for i in range(1, 9)]
phoneme_cols = ['start', 'end', 'phoneme']
spec = MelSpectrogram(n_fft=1600, f_max=20000)
padding = ConstantPad1d((400, 400), 0)
epsilon = 1e-6


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
                            phoneme_samples[dataset][dialect][row[2]].append(
                                torch.log(spec(subsample).mean(2) + epsilon)
                            )

                for phoneme in phoneme_samples[dataset][dialect]:
                    phoneme_samples[dataset][dialect][phoneme] = \
                        torch.cat(phoneme_samples[dataset][dialect][phoneme]).data.numpy()

    with open(pkl_path, 'wb') as f:
        pickle.dump(phoneme_samples, f)


def generate_data_loaders(dialect, pkl_path='timit_tokenized.pkl', batch_size=1):
    with open(pkl_path, 'rb') as f:
        data_pkl = pickle.load(f)

    train_data = torch.cat([torch.from_numpy(x) for x in data_pkl['TRAIN'][dialect].values()])
    test_data = torch.cat([torch.from_numpy(x) for x in data_pkl['TEST'][dialect].values()])

    return (
        DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True
        ),

        DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=True
        )
    )


if __name__ == '__main__':
    preprocess()
