import pickle

path=r"Y:\code\raw_to_classification\data\scalingAndFolding\FeaturesChannels@30@prep-defaultprep\noScaling_all\folds-noScaling@all.pkl"
pickle_file = open(path, 'rb')
data = pickle.load(pickle_file)
pickle_file.close()

data.keys()
data[-1].keys()