import shutil
import glob
import os

# for i, fdr in enumerate(os.listdir("../train")):
#     os.rename(f"../train/{fdr}", f"../train/batch{i}")

for i, fdr in enumerate(os.listdir("../train")):
    files_within = glob.glob(f"../train/{fdr}/*_relief.jpg")
#     # # print(fdr/)
    print(fdr, len(files_within))
#     os.rename(fdr, f"../train/batch{i}")