import os
import pickle

spectypes = ["true", "annulus", "annulus_error"]
fwhms = ["1pix", "3pix", "10pix"]

for spectype in spectypes:
    for fwhm in fwhms:
        fname = f"../data/HI_{spectype}_spectra_{fwhm}.pkl"
        dirname = fname.replace(".pkl", "")
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        resdirname = dirname.replace("/data/", "/results/")
        if not os.path.isdir(resdirname):
            os.mkdir(resdirname)

        with open(fname, "rb") as f:
            data = pickle.load(f)

        for idx in range(len(data["data_list"])):
            print(fname, idx)

            datum = {}
            for key in data.keys():
                datum[key] = data[key][idx]

            with open(f"{dirname}/{idx}.pkl", "wb") as f:
                pickle.dump(datum, f)
