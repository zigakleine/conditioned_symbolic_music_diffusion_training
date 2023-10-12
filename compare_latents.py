import pickle
import numpy as np

original_song_path = "./0*+0*p1-p2-tr-no.pkl"
original_song = pickle.load(open(original_song_path, "rb"))
original_song = original_song[0]

training_song_path = "./0*+0*p1-p2-tr-no_loc.pkl"
training_song = pickle.load(open(training_song_path, "rb"))
training_song = training_song[0]

print(np.sum(training_song - original_song))