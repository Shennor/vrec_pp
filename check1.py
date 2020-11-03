from utils import find_files, ensure_dir_for_filename, ensures_dir, load_npy
from audio import read_mfcc, make_mfcc_from_frames, extract_speaker_and_utterance_ids
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
import numpy as np
import os
import pyaudio
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

AUDIOBASE = 'student_samples/'
PREDICTED_BASE = 'student_features/'

def make_student_prediction(checkpoint):
    students = find_files(AUDIOBASE)
    model = DeepSpeakerModel()
    model.m.load_weights(checkpoint, by_name=True)
    ensures_dir(PREDICTED_BASE)
    for stud in students:
        ensure_dir_for_filename(stud)
        mfcc = sample_from_mfcc(read_mfcc(stud, SAMPLE_RATE), NUM_FRAMES)
        predict = model.m.predict(np.expand_dims(mfcc, axis=0))
        sp, utt = extract_speaker_and_utterance_ids(stud)
        utt = utt.split('.wav')[-2]
        filename = os.path.join(PREDICTED_BASE, f'{sp}/{utt}.npy')
        ensure_dir_for_filename(filename)
        np.save(filename, predict)

def predict(checkpoint, filename):
    model = DeepSpeakerModel()
    model.m.load_weights(checkpoint, by_name=True)
    return model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc(filename, SAMPLE_RATE), NUM_FRAMES), axis=0))

def sort_by_average(base):
    list_d = list(base.items())
    #[speaker, [average, max_cosine]]
    #sort >= by average
    list_d.sort(key=lambda i: i[1][0], reverse=True)
    return list_d

def sort_by_max_cosine(base):
    list_d = list(base.items())
    list_d.sort(key=lambda i: i[1][1], reverse=True)
    return list_d

def print_statistics(base):
    sorted_average = sort_by_average(base)
    sorted_max_cosine = sort_by_max_cosine(base)
    print('Average probability')
    for i in sorted_average:
        percent = i[1][0] * 100
        print(f'Speaker {i[0]}: {percent}%')
    print('Max match')
    for i in sorted_max_cosine:
        percent = i[1][1] * 100
        print(f'Speaker {i[0]}: {percent}%')

def predict_by_micro(checkpoint):
    p = pyaudio.PyAudio()
    for i in range (p.get_device_count()):
        print(i, p.get_device_info_by_index(i)['name'])
        chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1 
    rate = SAMPLE_RATE
    seconds = 3
    filename = 'output_sound.wav'
    p = pyaudio.PyAudio()
    print('Recording...')
    
    stream = p.open(format=sample_format,
    channels=channels,
    rate=rate,
    frames_per_buffer=chunk,
    input_device_index=14, # индекс устройства с которого будет идти запись звука 
    input=True)

    frames = [] # Инициализировать массив для хранения кадров

    # Хранить данные в блоках в течение 3 секунд
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        data = np.frombuffer(data)
        #print(type(data))
        frames.append(data)
    # Остановить и закрыть поток
    print(type(frames[1]))
    frames = np.array(frames)
    
    stream.stop_stream()
    stream.close()
    # Завершить интерфейс PortAudio
    p.terminate()
    print('Finished recording!')
    model = DeepSpeakerModel()
    model.m.load_weights(checkpoint, by_name=True)
    return model.m.predict(np.expand_dims(sample_from_mfcc(make_mfcc_from_frames(frames, SAMPLE_RATE), NUM_FRAMES), axis=0))

    

def find_ids(pred_tensor):
    samples = find_files(PREDICTED_BASE, 'npy')
    base = {'none': [0, -1, -1]}
    #{speaker: [cosine sum, number of utterances, max cosine]}
    for sample in samples:
        ensure_dir_for_filename(sample)
        sp = sample.split('/')[-2]
        #print(sp)
        if not(sp in base):
            base[sp] = [0, 0, -1]
        samp_tensor = load_npy(sample)
        cos = batch_cosine_similarity(pred_tensor, samp_tensor)
        base[sp][0] += cos;
        base[sp][1] += 1
        if(cos > base[sp][2]):
            base[sp][2] = cos
    #max_cos_key = 'none'
    #max_cos_val = -1
    #max_average_key = 'none'
    #max_average_val = -1
    res = dict()
    #{speaker: [average, max_cosine]}
    for key in base:
        average = base[key][0]/base[key][1]
        res[key] = [average, base[key][2]]
        #if average > max_average_val:
        #    max_average_val = average
        #    max_average_key = key
        #if base[key][2] > max_cos_val:
        #    max_cos_val = base[key][2]
        #    max_cos_key = key
    #if max_cos_val < 0.65:
    #    res = "none"
    #elif max_cos_val > 0.9:
    #    res = max_cos_key
    #else:
    #    res = max_average_key
    return res

checkpoint = 'ResCNN_checkpoint_36.h5'
#make_student_prediction(checkpoint)
filename = input('Enter filename to predict >>')
pred_tensor = predict(checkpoint, filename)
#pred_tensor = predict_by_micro(checkpoint)
base = find_ids(pred_tensor)
print_statistics(base)
