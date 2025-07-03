# Neural Operators for Acoustic Diffusers

1. Activate venv: `conda activate diffusers`
2. Run pipeline: `python run.py --dir mini_imagenet/n13054560 --csv_file mini_imagenet/coeffs.csv` -> loads all pictures from `mini_imagenet/n13054560` and writes the diffusion coefficients per frequency to `mini_imagenet/coeffs.csv`