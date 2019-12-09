import torch
from model.vanillavae.joint_vaes import VAE, SharedEncoder


def latent_space_interpolator(encoder_mlp, decoder_vae, mspec_lst, z_coeff):

    # Some sanity checks
    assert len(mspec_lst) == len(z_coeff)
    assert sum(z_coeff) == 1
    assert isinstance(encoder_mlp, SharedEncoder)
    assert isinstance(decoder_vae, VAE)

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


if __name__ == '__main__':
    # Example usage
    output_spec = latent_space_interpolator(
        encoder_mlp=SharedEncoder(),
        decoder_vae=VAE(),
        mspec_lst=[torch.randn(128), torch.randn(128)],
        z_coeff=[0.4, 0.6]
    )
    print(output_spec)
