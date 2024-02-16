from linter.criterions.lpips_with_discriminator import LPIPS_with_discriminator


def get_loss(disc_start, kl_weight, disc_weight):
    return LPIPS_with_discriminator(
        disc_start=disc_start, kl_weight=kl_weight, disc_weight=disc_weight
    )
