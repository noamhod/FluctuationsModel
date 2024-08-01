import pickle
# fpkl = "/Users/noamtalhod/tmp/root/rootslice_E10_dL10.pkl"
fpkl = "scan_example.pkl"
with open(fpkl,'rb') as handle:
    data = pickle.load(handle)
    print(data)
