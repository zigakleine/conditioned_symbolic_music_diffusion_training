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
                    song_rel_urls = song["encoded_song_urls"]
                    for song_rel_url in song_rel_urls:
                        for i in range(song["num_sequences"]):
                            sequence = {"url": song_rel_url, "index": i, "genre": genre, "composers": composers}
                            self.all_nesmdb_metadata.append(sequence)

    def __getitem__(self, index):
        enc_seq_rel_path = self.all_nesmdb_metadata[index]["url"]
        enc_seq_abs_path = os.path.join(self.current_dir, enc_seq_rel_path)

        enc_seq = pickle.load(open(enc_seq_abs_path, "rb"))
        enc_seq = enc_seq[self.all_nesmdb_metadata[index]["index"]]

        genre = self.all_nesmdb_metadata[index]["genre"]
        composers = self.all_nesmdb_metadata[index]["composers"]

        label_choice = np.random.choice([i for i in range(len(composers))], 1)
        composer = composers[label_choice[0]]

        if self.transform:
            enc_seq = self.transform(enc_seq, self.min_max["min"], self.min_max["max"])
        #
        #{"g":-1, "c":-1}
        return enc_seq, [genre, composer]

    def __len__(self):
        return len(self.all_nesmdb_metadata)
