from run_dip_denoise_deblur import *

dataset = BSDS300Dataset(use_patches=False)

run_method(dataset, dataset_name="BSDS300", task="denoise", method="ADMM-DIP-COMPILE", fsavepath="../results", verbose=True, sigma=0.15)