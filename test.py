# import importlib
# import run_dip_denoise_deblur as dip_runner
# from IPython.display import display, Image as NotebookImage

# dip_runner = importlib.reload(dip_runner)

# dataset = dip_runner.BSDS300Dataset(use_patches=False)

# comparison = dip_runner.compare_admm_tv_methods(
#     dataset,
#     image_index=0,
#     sigma=0.1,
#     fsavepath="../results",
#     verbose=True,
#     noise_seed=1,
#     model_seed=1,
# )

# display(NotebookImage(filename=comparison["montage_path"]))
# display(NotebookImage(filename=comparison["curves_path"]))



from run_dip_denoise_deblur import *

dataset = BSDS300Dataset(use_patches=False)
run_method(dataset, dataset_name="BSDS300", task="denoise", method= "ADMM-DIP", fsavepath="../results", verbose=True, sigma=0.1)

