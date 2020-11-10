from utils import find_files, ensure_dir_for_filename, ensures_dir, load_npy
from audio import read_mfcc, extract_speaker_and_utterance_ids
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
import numpy as np
import os
import pyaudio
import wave
import sounddevice as sd
import re
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

from PySimpleGUI import cprint

AUDIOBASE = 'student_samples/'
PREDICTED_BASE = 'student_features/'

def make_student_prediction( model):
    students = find_files(AUDIOBASE)
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


def predict_by_file(filename, model):
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
    cprint('Average probability')
    for i in sorted_average:
        percent = i[1][0] * 100
        cprint(f'Speaker {i[0]}: {percent}%')
    cprint('Max match')
    for i in sorted_max_cosine:
        percent = i[1][1] * 100
        cprint(f'Speaker {i[0]}: {percent}%')


def get_device_list():
    p = pyaudio.PyAudio()
    res = list()
    for i in range (p.get_device_count()):
        res.append(str(i) + ' ' + str(p.get_device_info_by_index(i)['name']))
    return res


def predict_by_id(device_id, model):
    p = pyaudio.PyAudio()
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1 
    rate = SAMPLE_RATE
    seconds = 4
    filename = 'last_input_voice.wav'
    stream = p.open(format=sample_format,
    channels=channels,
    rate=rate,
    frames_per_buffer=chunk,
    input_device_index=device_id, 
    input=True)

    frames = [] 
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc('last_input_voice.wav', SAMPLE_RATE), NUM_FRAMES), axis=0))


def predict_default(model):    
    p = pyaudio.PyAudio()
    speakers = p.get_default_output_device_info()["hostApi"]
    #speakers = sd.query_devices(kind='output')['hostapi']
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    print(p.get_default_output_device_info()["defaultSampleRate"])
    print(SAMPLE_RATE)
    #rate = int(p.get_default_output_device_info()["defaultSampleRate"])
    rate = SAMPLE_RATE
    seconds = 4
    filename = 'last_input_voice.wav'
    stream = p.open(format=sample_format,
        channels=channels,
        rate=rate,
        frames_per_buffer=chunk,
        #input_host_api_specific_stream_info=speakers, 
        input_device_index=speakers,
        input=True)

    frames = [] 
    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc('last_input_voice.wav', SAMPLE_RATE), NUM_FRAMES), axis=0))


def id_list():
    samples = find_files(PREDICTED_BASE, 'npy')
    base = set()
    for sample in samples:
        sp = sample.split('/')[-2]
        base.add(sp)
    return list(sorted(base))


def find_statistics(pred_tensor, basepath):
    samples = find_files(basepath, 'npy')
    base = dict()
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
    res = dict()
    #{speaker: [average cosine, max cosine]}
    for key in base:
        average = base[key][0]/base[key][1]
        res[key] = [average, base[key][2]]
    return res


def id_exist(student_id):
    if len(student_id) == 0:
        return False
    elif not os.path.exists(PREDICTED_BASE + student_id):
        return False
    else:
        return True
    

def verify_student(pred_tensor, student_id):
    student_id = re.sub(r'\s', '', student_id) #Уберем все пробельные символы
    student_id.replace('/', '')
    if not id_exist(student_id): 
        return dict()
    else:
        return find_statistics(pred_tensor,PREDICTED_BASE + student_id)


def print_verification_result(base):
    if len(base) == 0:
        cprint("ID is empty or doesn't exist...")
    else:
        #print(type(base))
        key = list(base)[0]
        #print(base)
        cprint(f'Verification result for {key}:')
        cprint(f'Average similarity: {base[key][0] * 100}%')
        cprint(f'Max similarity: {base[key][1] * 100}%')

#Model start:
#model = DeepSpeakerModel()
#checkpoint = 'ResCNN_checkpoint_36.h5'
#model.m.load_weights(checkpoint, by_name=True)

#student prediction:
#make_student_prediction(model)

#predict by file:
#filename = input('Enter filename to predict >> ')
#pred_tensor = predict(filename, model)

#predict by device:
#device_id = int(input('Enter device id >> '))
#pred_tensor = predict_by_micro(device_id, model)

#identification:
#base = find_statistics(pred_tensor, PREDICTED_BASE)
#print_statistics(base)

#verification:
#student_id = input('Enter student ID to verify >> ')
#base = verify_student(pred_tensor, student_id)
#print_verification_result(base)
