
import numpy as np
import os
import tensorflow.compat.v1 as tf
import time

# from google.colab import files

# import magenta.music as mm
# from magenta.music.sequences_lib import concatenate_sequences
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

from note_seq import midi_io, midi_file_to_note_sequence
from note_seq import sequences_lib

tf.disable_v2_behavior()
print('Done!')


import ctypes
import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo


# NESMDB--> ---Define the struct to match the C++ side
class nesmdb_sequence_array(ctypes.Structure):
    _fields_ = [("sequence", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))),
                ("dim1", ctypes.c_int),
                ("dim2", ctypes.c_int),
                ("dim3", ctypes.c_int)]

class lakh_sequence_array(ctypes.Structure):
    _fields_ = [("sequences", ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_int))))),
                ("dim1", ctypes.c_int),
                ("dim2", ctypes.c_int),
                ("dim3", ctypes.c_int),
                ("dim4", ctypes.c_int),
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

        if db_type == "nesmdb":
            self.sequence_array = nesmdb_sequence_array
        elif db_type == "lakh":
            self.sequence_array = lakh_sequence_array
        else:
            self.sequence_array = None

        # Define the return type of the function
        self.cpp_lib.extract_note_sequences_from_midi.restype = self.sequence_array

    def song_from_midi_nesmdb(self, file_path, transposition, transposition_plus):

        midi_file_loc = ctypes.c_char_p(file_path.encode())
        transposition_cpp = ctypes.c_int(transposition)
        transposition_plus_cpp = ctypes.c_bool(transposition_plus)

        # Call the C++ function
        sequence_array = self.cpp_lib.extract_note_sequences_from_midi(midi_file_loc, transposition_cpp, transposition_plus_cpp)

        # Convert the array to a NumPy array
        sequences_ = np.zeros((sequence_array.dim1, sequence_array.dim2, sequence_array.dim3), dtype=np.int32)

        for i in range(sequence_array.dim1):
            for j in range(sequence_array.dim2):
                for k in range(sequence_array.dim3):
                    sequences_[i, j, k] = sequence_array.sequence[i][j][k]

        # for i in range(len(sequences_)):
        #     print("measure-" + str(i))
        #     for j in range(len(sequences_[i])):
        #         for k in range(len(sequences_[i][j])):
        #             print(str(sequences_[i][j][k]), end=" ")
        #         print("")
        #     print("\n")
        # print("\n")

        return sequences_

    def song_from_midi_lakh(self, file_path):

        midi_file_loc = ctypes.c_char_p(file_path.encode())

        # Call the C++ function
        sequences_array = self.cpp_lib.extract_note_sequences_from_midi(midi_file_loc)

        if sequences_array.successful == 0:
            return None

        # Convert the array to a NumPy array
        sequences_ = np.zeros((sequences_array.dim1, sequences_array.dim2, sequences_array.dim3, sequences_array.dim4), dtype=np.int32)

        for h in range(sequences_array.dim1):
            for i in range(sequences_array.dim2):
                for j in range(sequences_array.dim3):
                    for k in range(sequences_array.dim4):
                        sequences_[h, i, j, k] = sequences_array.sequences[h][i][j][k]

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

        dim1, dim2, dim3 = song_data.shape

        event_sequences = []
        for i in range(dim1):
            measure = []
            for j in range(dim2):
                track = []
                for k in range(dim3):
                    current_val = song_data[i, j, k]

                    if current_val > -1:
                        if 0 <= current_val < 128:
                            track.append((1, current_val))
                        elif 128 <= current_val < 256:
                            track.append((2, current_val - 128))
                        elif 256 <= current_val < 352:
                            track.append((3, current_val - 256 + 1))
                        elif 352 <= current_val < 360:
                            track.append((4, current_val - 352 + 1))
                        elif 360 <= current_val < 488:
                            track.append((0, current_val - 360 + 1))
                        elif current_val == 489:
                            track.append((5, 0))
                measure.append(track)
            event_sequences.append(measure)

        for i in range(len(event_sequences)):
            print("measure-" + str(i))
            for j in range(len(event_sequences[i])):
                for k in range(len(event_sequences[i][j])):
                    print(str(event_sequences[i][j][k]), end=" ")
                print("")
            print("\n")
        print("\n")

        mid = MidiFile()

        measure_ticks = 96
        nes_tracks = 4
        ticks_per_beat = 480  # Standard MIDI ticks per beat

        time_shift_multiplier = ticks_per_beat / (measure_ticks/4)  # Adjust to match the time-shift event range

        track_metadata = MidiTrack()
        mid.tracks.append(track_metadata)
        track_metadata.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        track_metadata.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))
        track_metadata.append(MetaMessage('time_signature', numerator=4, denominator=1, time=int(len(event_sequences)*measure_ticks*time_shift_multiplier)))
        track_metadata.append(MetaMessage('end_of_track', time=0))

        ins_names = ['p1', 'p2', 'tr', 'no']
        ticks_from_last_event = [0, 0, 0, 0]
        for i in range(nes_tracks):
            track = MidiTrack()
            mid.tracks.append(track)
            track.append(MetaMessage('track_name', name=ins_names[i]))

        for s, sequence in enumerate(event_sequences):

            for t, seq_track in enumerate(sequence):
                ticks_passed = 0

                active_notes = []

                for event_type, event_value in seq_track:

                    if event_type == 1:  # Note-on event
                        mid.tracks[t+1].append(Message('note_on', note=event_value, velocity=3, time=ticks_from_last_event[t]))
                        ticks_from_last_event[t] = 0
                        active_notes.append(event_value)

                    elif event_type == 2:  # Note-off event
                        mid.tracks[t+1].append(Message('note_off', note=event_value, velocity=3, time=ticks_from_last_event[t]))
                        ticks_from_last_event[t] = 0
                        if event_value in active_notes:
                            active_notes.remove(event_value)

                    elif event_type == 3:  # Time-shift event
                        time_shift = int(event_value * time_shift_multiplier)
                        ticks_passed += time_shift

                        if ticks_passed > int(measure_ticks*time_shift_multiplier):
                            ticks_passed -= time_shift
                            break
                        else:
                            ticks_from_last_event[t] += time_shift

                # print(s, t, "-ticks_passed", ticks_passed/time_shift_multiplier)

                ticks_from_last_event[t] += int(measure_ticks*time_shift_multiplier) - ticks_passed

                for note in active_notes:
                    mid.tracks[t+1].append(
                        Message('note_off', note=note, velocity=3, time=ticks_from_last_event[t]))
                    ticks_from_last_event[t] = 0


                # print("")


        for i in range(nes_tracks+1):
            mid.tracks[i].append(MetaMessage('end_of_track', time=0))

            if mid.tracks[i].name == "no":
                for event in mid.tracks[i]:
                    if event.type == "note_on" or event.type == "note_off":
                        event.note = event.note // 8


        return mid


class multitrack_vae:

    def __init__(self, model_file_path, batch_size):
        self.config = configs.CONFIG_MAP['hier-multiperf_vel_1bar_med']
        self.model = TrainedModel(
            self.config, batch_size=batch_size,
            checkpoint_dir_or_path=model_file_path)
        self.model._config.data_converter._max_tensors_per_input = None

    def encode_sequence(self, song_data):

        num_measures = len(song_data)
        num_tracks = 4
        max_events = 64

        max_events_all = max_events*8
        num_tokens = 490

        inputs = []
        lengths = []
        controls = []

        for i in range(num_measures):
            one_hot_matrix = np.zeros((max_events_all, num_tokens), dtype=bool)
            lengths_measure = np.zeros((8, ), dtype=np.int32)
            controls_measure = np.empty((1, 0), dtype=np.float64)

            for j in range(num_tracks):
                for k in range(max_events):
                    current_value = song_data[i, j, k]
                    if current_value > -1:
                        lengths_measure[j] += 1
                        one_hot_matrix[max_events*j + k, current_value] = 1

            for l in range(8-num_tracks):
                one_hot_matrix[(l+4)*max_events, 489] = 1
                lengths_measure[l+4] += 1

            inputs.append(one_hot_matrix)
            lengths.append(lengths_measure)
            controls.append(controls_measure)

        self.model.sample()
        # start_time = time.time()
        z, _, _ = self.model.encode_tensors(inputs, lengths, controls)
        # end_time = time.time()
        # print("time in s:", (end_time - start_time), z)
        return z

    def decode_sequence(self, z, total_steps, temperature):

        # dec = self.model.decode(z, total_steps, temperature)
        song_tensors = self.model.decode_to_tensors(z, total_steps, temperature, None)

        num_measures = len(song_tensors)
        tracks_num = 4
        max_events = 64

        song_data = np.zeros((num_measures, tracks_num, max_events), dtype=np.int32)
        song_data.fill(-1)

        for i in range(num_measures):
            for j in range(tracks_num):
                for k in range(max_events):
                    token = np.argmax(song_tensors[i][(j * 64 + k)])
                    song_data[i, j, k] = token
                    if token == 489:
                        break

        return song_data


if __name__ == "__main__":
    pass

    # current_dir = os.getcwd()
    # model_rel_path = "multitrack_vae_model/model_fb256.ckpt"
    # batch_size = 32
    # total_steps = 512
    # model_path = os.path.join(current_dir, model_rel_path)
    # vae = multitrack_vae(model_path, batch_size)
    # temperature = 0.2
    # seqs = vae.model.sample(n=batch_size, length=total_steps, temperature=temperature)

    #-------------------

    # mario_file_path = "/Users/zigakleine/Desktop/conditioned_symbollic_music_diffusion_preprocessing/lmd_full/0/0cad5284ec963f245059ef42230c6e63.mid"
    #
    # current_dir = os.getcwd()
    # model_rel_path = "multitrack_vae_model/model_fb256.ckpt"
    # nesmdb_shared_library_rel_path = "ext_nseq_lakh_lib.so"
    # db_type = "lakh"
    #
    # batch_size = 32
    # temperature = 0.2
    # total_steps = 512
    #
    # model_path = os.path.join(current_dir, model_rel_path)
    # nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)
    #
    # db_proc = db_processing(nesmdb_shared_library_path, db_type)
    # song_data = db_proc.song_from_midi_lakh(mario_file_path)
    #
    # if song_data is not None:
    #     vae = multitrack_vae(model_path, batch_size)
    #
    #     song_data_reshaped = song_data.reshape(song_data.shape[0]*song_data.shape[1], 4, 64)
    #     z = vae.encode_sequence(song_data_reshaped)
    #
    #     new_song_data = vae.decode_sequence(z, total_steps, temperature)
    #
    #     midi = db_proc.midi_from_song(new_song_data)
    #
    #     midi.save("new_song.mid")

    #-------------------

    #
    # mario_file_path = "/Users/zigakleine/Desktop/conditioned_symbollic_music_diffusion_preprocessing/nesmdb_flat/322_SuperMarioBros__00_01RunningAbout.mid"
    #
    # current_dir = os.getcwd()
    # model_rel_path = "multitrack_vae_model/model_fb256.ckpt"
    # nesmdb_shared_library_rel_path = "ext_nseq_nesmdb_lib.so"
    # db_type = "nesmdb"
    #
    # batch_size = 32
    # temperature = 0.2
    # total_steps = 512
    #
    # transposition = 0
    # transposition_plus = True
    #
    # model_path = os.path.join(current_dir, model_rel_path)
    # nesmdb_shared_library_path = os.path.join(current_dir, nesmdb_shared_library_rel_path)
    #
    # db_proc = db_processing(nesmdb_shared_library_path, db_type)
    # vae = multitrack_vae(model_path, batch_size)
    #
    # song_data = db_proc.song_from_midi_nesmdb(mario_file_path, transposition, transposition_plus)
    # song_data_reshaped = song_data[:, 1:, :]
    # z = vae.encode_sequence(song_data_reshaped)
    #
    # new_song_data = vae.decode_sequence(z, total_steps, temperature)
    #
    # midi = db_proc.midi_from_song(new_song_data)
    #
    # midi.save("new_song.mid")
