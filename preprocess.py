import pandas
import pickle
import librosa
import torch
import torchaudio

from collections import defaultdict
from glob import glob
from torch.nn import ConstantPad1d
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram

n_fft = 1600
n_mels = 128
f_max = 20000
sr = 16000
hop_length = 160
win_length = hop_length * 2
max_len = sr // 2
dialects = ['DR' + str(i) for i in range(1, 9)]
phoneme_cols = ['start', 'end', 'phoneme']
spec = MelSpectrogram(n_fft=n_fft, f_max=f_max)
epsilon = 1e-6


def get_mspec(y):
    return librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def inverse_melspec(s):
    return librosa.feature.inverse.mel_to_audio(s, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def wav_to_padded_mspec_flat_tensor(wav, length):
    assert(length <= max_len)
    p_len = max_len - length
    padding = ConstantPad1d((0, p_len), 0)
    return torch.Tensor(get_mspec(padding(wav).flatten().data.numpy())).view(-1)


# to get stitched audio from a list of phoneme wavs `wavs`, run
# `librosa.output.write_wav('outpath.wav', numpy.hstack(wavs), sr=sr)`
def padded_mspec_flat_tensor_to_wav(mspec_flat, orig_len):
    padded_mspec = mspec_flat.view(n_mels, -1).data.numpy()
    return inverse_melspec(padded_mspec)[:orig_len]


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
                            phn_len = row[1] - row[0]
                            if phn_len > max_len:
                                continue
                            mspec_flat_tensor = wav_to_padded_mspec_flat_tensor(subsample, phn_len)
                            phoneme_samples[dataset][dialect][row[2]].append(
                                torch.log(mspec_flat_tensor + epsilon)
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
