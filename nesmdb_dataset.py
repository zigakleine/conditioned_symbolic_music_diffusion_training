import os
import torch
from torch.utils.data import Dataset
import pickle
import json
import numpy as np


class NesmdbMidiDataset(Dataset):

    def __init__(self, min_max=None, transform=None):

        categories_path = "./db_metadata/nesmdb/nesmdb_categories.pkl"
        categories_indices = pickle.load(open(categories_path, "rb"))

        self.transform = transform
        self.min_max = min_max

        self.metadata_folder = "db_metadata"
        self.database_folder = "nesmdb"
        self.current_dir = os.getcwd()
        self.encoded_dir = "/storage/local/ssd/zigakleine-workspace"
        # self.encoded_dir = os.getcwd()
        self.all_nesmdb_metadata = []
        self.metadata_filename = "nesmdb_updated2808.pkl"
        nesmdb_metadata_abs_path = os.path.join(self.current_dir, self.metadata_folder, self.database_folder,
                                                self.metadata_filename)
        metadata = pickle.load(open(nesmdb_metadata_abs_path, "rb"))

        for game in metadata:
            composers = [categories_indices["composers"][composer] for composer in metadata[game]["composers"]]
            genre = categories_indices["genres"][metadata[game]["genre"]]
            for song in metadata[game]["songs"]:
                if song["is_encodable"]:

                    emotion_q = categories_indices["emotions"][song["emotion_pred_same_vel"]]
                    song_rel_urls = song["encoded_song_urls"]
                    for song_rel_url in song_rel_urls:
                        for i in range(song["num_sequences"]):
                            sequence = {"url": song_rel_url, "index": i, "genre": genre, "composers": composers, "emotion": emotion_q}
                            self.all_nesmdb_metadata.append(sequence)

    def __getitem__(self, index):
        enc_seq_rel_path = self.all_nesmdb_metadata[index]["url"]
        # enc_seq_abs_path = os.path.join(self.current_dir, enc_seq_rel_path)
        enc_seq_abs_path = os.path.join(self.encoded_dir, enc_seq_rel_path)

        enc_seq = pickle.load(open(enc_seq_abs_path, "rb"))
        enc_seq = enc_seq[self.all_nesmdb_metadata[index]["index"]]

        enc_seq_tracks = np.split(enc_seq, 4, axis=0)
        enc_seq_hstacked = np.hstack(enc_seq_tracks)

        # genre = self.all_nesmdb_metadata[index]["genre"]
        # composers = self.all_nesmdb_metadata[index]["composers"]
        emotion = self.all_nesmdb_metadata[index]["emotion"]

        # label_choice = np.random.choice([i for i in range(len(composers))], 1)
        # composer = composers[label_choice[0]]

        if self.transform:
            enc_seq_hstacked = self.transform(enc_seq_hstacked, -14., 14.)
        #
        #{"g":-1, "c":-1}
        # return enc_seq_hstacked, [genre, composer]
        return enc_seq_hstacked, emotion

    def __len__(self):
        return len(self.all_nesmdb_metadata)
