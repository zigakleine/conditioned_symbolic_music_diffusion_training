
import numpy as np
import os
import tensorflow.compat.v1 as tf
# import time
# from scipy.io import wavfile

from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
# import note_seq

# from note_seq import midi_io, synthesize, play_sequence
# from note_seq.sequences_lib import concatenate_sequences

tf.disable_v2_behavior()
print('Done!')


import ctypes
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo


# NESMDB--> ---Define the struct to match the C++ side
class nesmdb_sequence_array(ctypes.Structure):
    _fields_ = [("sequence", ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
                ("dim1", ctypes.c_int),
                ("dim2", ctypes.c_int)]

class lakh_sequence_array(ctypes.Structure):
    _fields_ = [("sequences", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))),
                ("dim1", ctypes.c_int),
                ("dim2", ctypes.c_int),
                ("dim3", ctypes.c_int),
                ("successful", ctypes.c_int)]


def check_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print("GPU:", gpu)
            print("Name:", gpu.name)
            # print("Memory:", gpu.memory_limit)
            print("Device Type:", gpu.device_type)
            print("")

        # Allowing TensorFlow to use memory dynamically on a GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPUs found. TensorFlow will use CPU.")


class db_processing:

    def __init__(self, shared_library_path, db_type):

        # Load the shared library
        self.cpp_lib = ctypes.CDLL(shared_library_path)

        if db_type == "nesmdb_singletrack":
            self.sequence_array = nesmdb_sequence_array
        elif db_type == "lakh_singletrack":
            self.sequence_array = lakh_sequence_array
        else:
            self.sequence_array = None

        # Define the return type of the function
        self.cpp_lib.extract_note_sequences_from_midi_singletrack.restype = self.sequence_array

    def song_from_midi_nesmdb(self, file_path, transposition, transposition_plus):

        midi_file_loc = ctypes.c_char_p(file_path.encode())
        transposition_cpp = ctypes.c_int(transposition)
        transposition_plus_cpp = ctypes.c_bool(transposition_plus)

        # Call the C++ function
        sequence_array = self.cpp_lib.extract_note_sequences_from_midi_singletrack(midi_file_loc, transposition_cpp, transposition_plus_cpp)

        # Convert the array to a NumPy array
        sequences_ = np.zeros((sequence_array.dim1, sequence_array.dim2), dtype=np.int32)

        for i in range(sequence_array.dim1):
            for j in range(sequence_array.dim2):

                current_value = sequence_array.sequence[i][j]
                if 21 <= current_value <= 108:
                    sequences_[i, j] = current_value - 19
                elif current_value == -2:
                    sequences_[i, j] = 1
                else:
                    sequences_[i, j] = 0

                if i == (sequence_array.dim1 - 1) and current_value >= 0:
                    if current_value >= 16:
                        current_value = current_value // 8
                    sequences_[i][j] = current_value + 2

        for i in range(len(sequences_)):
            for j in range(len(sequences_[i])):
                current_element = sequences_[i][j]
                if j == 0:
                    prev_element = 0
                else:
                    prev_element = sequences_[i][j-1]

                if prev_element == 0 and current_element == 1:
                    sequences_[i][j] = 0

            #     print(str(sequences_[i][j]), end=" ")
            # print("")
            # print("\n")

        sequences_ = sequences_[1:, :]
        return sequences_

    def song_from_midi_lakh(self, file_path):

        midi_file_loc = ctypes.c_char_p(file_path.encode())

        # Call the C++ function
        sequences_array = self.cpp_lib.extract_note_sequences_from_midi_singletrack(midi_file_loc)

        if sequences_array.successful == 0:
            return None

        # Convert the array to a NumPy array
        sequences_ = np.zeros((sequences_array.dim1, sequences_array.dim2, sequences_array.dim3), dtype=np.int32)

        for i in range(sequences_array.dim1):
            for j in range(sequences_array.dim2):
                for k in range(sequences_array.dim3):
                    current_value = sequences_array.sequences[i][j][k]

                    if 21 <= current_value <= 108:
                        sequences_[i, j, k] = current_value - 19
                    elif current_value == -2:
                        sequences_[i, j, k] = 1
                    else:
                        sequences_[i, j, k] = 0

                    if j == (sequences_array.dim2 - 1) and current_value >= 0:
                        if current_value >= 16:
                            current_value = current_value // 8
                        sequences_[i, j, k] = current_value + 2

        for i in range(len(sequences_)):
            for j in range(len(sequences_[i])):
                for k in range(len(sequences_[i][j])):
                    current_element = sequences_[i][j][k]
                    if k == 0:
                        prev_element = 0
                    else:
                        prev_element = sequences_[i][j][k-1]

                    if prev_element == 0 and current_element == 1:
                        sequences_[i][j][k] = 0

        # for h in range(sequences_.shape[0]):
        #     print("block-" + str(h))
        #     for i in range(sequences_.shape[1]):
        #         print("measure-" + str(i))
        #         for j in range(sequences_.shape[2]):
        #             for k in range(sequences_.shape[3]):
        #                 print(str(sequences_[h, i, j, k]), end=" ")
        #             print("")
        #         print("\n")
        #     print("\n")
        #     print("\n")
        return sequences_


    def midi_from_song(self, song_data):

        mid = MidiFile()

        song_measures = int(song_data.shape[1]/16)
        nes_tracks = 4
        ticks_per_beat = 480  # Standard MIDI ticks per beat
        ticks_per_sixteenth = int(ticks_per_beat/4)


        track_metadata = MidiTrack()
        mid.tracks.append(track_metadata)
        track_metadata.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track_metadata.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))
        track_metadata.append(MetaMessage('time_signature', numerator=4, denominator=1, time=int(song_measures*ticks_per_sixteenth*16)))
        track_metadata.append(MetaMessage('end_of_track', time=0))

        ins_names = ['p1', 'p2', 'tr', 'no']
        for i in range(nes_tracks):
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage('track_name', name=ins_names[i]))

        for track_num in range(len(song_data)):
            ticks_from_last_event = 0
            note_playing = -1
            for sequence_timestep in range(len(song_data[track_num])):
                current_token = song_data[track_num, sequence_timestep]

                if current_token > 1:
                    if track_num == (len(song_data) - 1):
                        current_token -= 2
                    else:
                        current_token += 19

                if current_token == 0:
                    if not (note_playing == -1):
                        mid.tracks[track_num + 1].append(
                            Message('note_off', note=note_playing, velocity=3, time=ticks_from_last_event))
                        ticks_from_last_event = 0
                        note_playing = -1
                elif current_token == 1:
                    pass
                else:
                    if not (note_playing == -1):
                        mid.tracks[track_num + 1].append(
                            Message('note_off', note=note_playing, velocity=3, time=ticks_from_last_event))
                        ticks_from_last_event = 0
                        mid.tracks[track_num + 1].append(
                            Message('note_on', note=current_token, velocity=3, time=ticks_from_last_event))
                        ticks_from_last_event = 0
                        note_playing = current_token
                    else:
                        mid.tracks[track_num + 1].append(
                            Message('note_on', note=current_token, velocity=3, time=ticks_from_last_event))
                        ticks_from_last_event = 0
                        note_playing = current_token

                ticks_from_last_event += ticks_per_sixteenth

            if not (note_playing == -1):
                mid.tracks[track_num + 1].append(
                    Message('note_off', note=note_playing, velocity=3, time=ticks_from_last_event))
                ticks_from_last_event = 0

            mid.tracks[track_num+1].append(MetaMessage('end_of_track', time=ticks_from_last_event))

            if mid.tracks[track_num+1].name == "no":
                for event in mid.tracks[track_num+1]:
                    if event.type == "note_on" or event.type == "note_off":
                        if event.note > 15:
                            event.note = event.note // 8

        return mid


class singletrack_vae:

    def __init__(self, model_file_path, batch_size):
        self.config = configs.CONFIG_MAP['cat-mel_2bar_big']
        self.model = TrainedModel(
            self.config, batch_size=batch_size,
            checkpoint_dir_or_path=model_file_path)

    def encode_sequence(self, song_data):

        num_timesteps = song_data.shape[1]
        num_tracks = 4
        num_measures = num_timesteps//16

        token_vocab_size = 90
        num_tokens = 32

        inputs = []
        lengths = []
        controls = []

        for i in range(num_tracks):

            for j in range(num_timesteps):

                if j % num_tokens == 0:
                    one_hot_matrix = np.zeros((num_tokens, token_vocab_size), dtype=bool)
                    lengths_measure = 0
                    controls_measure = np.empty((1, 0), dtype=np.float64)

                current_value = song_data[i, j]
                one_hot_matrix[j % num_tokens, current_value] = 1
                lengths_measure += 1

                if (j+1) % num_tokens == 0:
                    inputs.append(one_hot_matrix)
                    lengths.append(lengths_measure)
                    controls.append(controls_measure)

            if not (num_timesteps % num_tokens == 0):
                inputs.append(one_hot_matrix)
                lengths.append(lengths_measure)
                controls.append(controls_measure)

        # self.model.sample()
        # start_time = time.time()
        z, _, _ = self.model.encode_tensors(inputs, lengths, controls)
        # end_time = time.time()
        # print("time in s:", (end_time - start_time), z)
        return z


    def decode_sequence(self, z, total_steps, temperature):

        # dec = self.model.decode(z, total_steps, temperature)
        song_tensors = self.model.decode_to_tensors(z, total_steps, temperature, None, False)

        tracks_num = 4
        num_double_measures = len(song_tensors)//tracks_num
        num_measures = num_double_measures * 2
        num_timesteps = num_measures * 16

        song_data = np.zeros((tracks_num, num_timesteps), dtype=np.int32)
        # song_data.fill(-1)

        for i in range(tracks_num):
            for j in range(num_timesteps):
                    token = np.argmax(song_tensors[num_double_measures * i + (j // total_steps)][j % total_steps])
                    song_data[i, j] = token


        return song_data

    def decode_sequence_full_results(self, z, total_steps, temperature):

        # dec = self.model.decode(z, total_steps, temperature)
        full_results = self.model.decode_to_tensors(z, total_steps, temperature, None, True)

        logits = full_results[1]
        song_tensors = full_results[2]
        tracks_num = 4
        num_double_measures = len(song_tensors)//tracks_num
        num_measures = num_double_measures * 2
        num_timesteps = num_measures * 16

        song_data = np.zeros((tracks_num, num_timesteps), dtype=np.int32)
        # song_data.fill(-1)

        for i in range(tracks_num):
            for j in range(num_timesteps):
                    token = np.argmax(song_tensors[num_double_measures * i + (j // total_steps)][j % total_steps])
                    song_data[i, j] = token


        return song_data, song_tensors, logits

# input_midi_path = "./twinkle.mid"
# input_midi = os.path.expanduser(input_midi_path)
# input = note_seq.midi_file_to_note_sequence(input_midi)
#
# z, _, _ = vae.model.encode([input])
# results = vae.model.decode(
#     length=vae.config.hparams.max_seq_len,
#     z=z,
#     temperature=temperature)
#
# note_seq.note_sequence_to_midi_file(results[0], "./twinkle_new.mid")
# print(results)
if __name__ == "__main__":

    mario_file_path = "/Users/zigakleine/Desktop/conditioned_symbollic_music_diffusion_preprocessing/nesmdb_flat/282_RoboWarrior_12_13EndingStaffRoll.mid"

    batch_size = 32
    temperature = 0.5
    total_steps = 32

    current_dir = os.getcwd()
    model_rel_path = "cat-mel_2bar_big.tar"
    model_path = os.path.join(current_dir, model_rel_path)
    db_type = "nesmdb_singletrack"
    # db_type = "lakh_singletrack"

    if db_type == "nesmdb_singletrack":

        nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_single_lib.so"

        transposition = 0
        transposition_plus = True

        model_path = os.path.join(current_dir, model_rel_path)
        nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)

        db_proc = db_processing(nesmdb_shared_library_path, db_type)
        vae = singletrack_vae(model_path, batch_size)

        song_data = db_proc.song_from_midi_nesmdb(mario_file_path, transposition, transposition_plus)

        z = vae.encode_sequence(song_data)
        song_data_ = vae.decode_sequence(z, total_steps, temperature)

        midi = db_proc.midi_from_song(song_data_)
        midi.save("mario_.mid")

    elif db_type == "lakh_singletrack":

        nesmdb_shared_library_rel_path = "ext_nseq_lakh_single_lib.so"


        model_path = os.path.join(current_dir, model_rel_path)
        lakh_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)

        db_proc = db_processing(lakh_shared_library_path, db_type)
        vae = singletrack_vae(model_path, batch_size)

        song_data = db_proc.song_from_midi_lakh(mario_file_path)

        z = vae.encode_sequence(song_data[0])
        song_data_ = vae.decode_sequence(z, total_steps, temperature)

        midi = db_proc.midi_from_song(song_data_)
        midi.save("mario_.mid")
