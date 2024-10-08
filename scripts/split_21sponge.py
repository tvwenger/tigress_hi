import os
import pickle
import numpy as np

dirname = "bighicat/21sponge"
if not os.path.isdir(dirname):
    os.mkdir(dirname)

with open("bighicat/21-SPONGE_em_abs_spectra.pkl", "rb") as f:
    data = pickle.load(f)

joint_data = {
    "data_list": [],
    "data_list_em": [],
    "x_values": [],
    "x_values_em": [],
    "errors": [],
    "errors_em": [],
    "source": [],
    "em_mom1": [],
    "em_mom2": [],
}

for source in data.keys():
    # calculate moments to inform velocity priors
    em_weights = data[source]["Tb"]
    em_mom1 = np.average(data[source]["v_em"], weights=em_weights)
    em_mom2 = np.average((data[source]["v_em"] - em_mom1) ** 2.0, weights=em_weights)

    # keep only non-nan absorption channels
    nan_abs = np.isnan(data[source]["emtau"])

    # save
    joint_data["data_list"].append(1.0 - data[source]["emtau"][~nan_abs])
    joint_data["data_list_em"].append(data[source]["Tb"])
    joint_data["x_values"].append(data[source]["v_abs"][~nan_abs])
    joint_data["x_values_em"].append(data[source]["v_em"])
    joint_data["errors"].append(data[source]["e_emtau"][~nan_abs])
    joint_data["errors_em"].append(data[source]["e_Tb"])
    joint_data["source"].append(source)
    joint_data["em_mom1"].append(em_mom1)
    joint_data["em_mom2"].append(em_mom2)

with open("bighicat/21-SPONGE_spectra.pkl", "wb") as f:
    pickle.dump(joint_data, f)

for idx in range(len(joint_data["data_list"])):
    datum = {}
    for key in joint_data.keys():
        datum[key] = joint_data[key][idx]

    with open(f"{dirname}/{idx}.pkl", "wb") as f:
        pickle.dump(datum, f)
