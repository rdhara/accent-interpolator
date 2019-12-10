import pickle
import torch
from torch.utils.data import DataLoader


def latent_space_interpolator(encoder_mlp, decoder_vae, mspec_lst, z_coeff):

    # Some sanity checks
    assert len(mspec_lst) == len(z_coeff)
    assert sum(z_coeff) == 1

    with torch.no_grad():

        # Make empty accumulators
        mu_interp = torch.zeros(decoder_vae.hidden_dim)
        logvar_interp = torch.zeros(decoder_vae.hidden_dim)

        # Weight the encoded outputs by the specified coefficients
        for i, spec in enumerate(mspec_lst):
            mu, logvar = encoder_mlp(spec)
            mu_interp += z_coeff[i] * mu
            logvar_interp += z_coeff[i] * logvar

        # Decode using the provided decoder (typically argmax of z_coeffs)
        z_interp = decoder_vae.reparameterize(mu_interp, logvar_interp)
        return decoder_vae.decode(z_interp)


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
