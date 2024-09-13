import pickle
fpkl = "output/slice_E10_dL10.pkl"
# fpkl = "scan_example.pkl"
with open(fpkl,'rb') as handle:
    data = pickle.load(handle)
    print(data)



