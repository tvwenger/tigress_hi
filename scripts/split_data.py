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
    with open(fname, "rb") as f:
        data = pickle.load(f)

    for idx in range(len(data["data_list"])):
        print(fname, idx)

        datum = {}
        for key in data.keys():
            datum[key] = data[key][idx]

        with open(f"{dirname}/{idx}.pkl", "wb") as f:
            pickle.dump(datum, f)
