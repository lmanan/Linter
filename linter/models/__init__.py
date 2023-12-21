from linter.models.autoencoder import AutoencoderKL

def get_model(dd_config, embed_dim ):
    return AutoencoderKL(dd_config,
                         embed_dim,
                         )