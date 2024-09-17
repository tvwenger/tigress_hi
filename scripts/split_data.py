import os
import pickle

fnames = [
    "../data/HI_joint_spectra.pkl",
    "../data/HI_joint_smooth_spectra.pkl",
]

for fname in fnames:
    dirname = fname.replace(".pkl", "")
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    for idx, row in data.iterrows():
        print(fname, idx)
        with open(f"{dirname}/{idx:06d}.pickle", 'wb') as f:
            pickle.dump(row, f)
