import numpy as np
import random

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity

def predict_id(checkpoint, audiofile):
    model = DeepSpeakerModel()
    model.m.load_weights(checkpoint, by_name=True)

    mfcc_utt = sample_from_mfcc(read_mfcc(audiofile, SAMPLE_RATE), NUM_FRAMES)
    predict_utt = model.m.predict(np.expand_dims(mfcc_utt, axis=0))

    mfcc_100_01 = sample_from_mfcc(read_mfcc('student_samples/100/01/100-049.wav', SAMPLE_RATE), NUM_FRAMES)
    print(type(mfcc_100_01))

    mfcc_100_02 = sample_from_mfcc(read_mfcc('student_samples/100/01/100-001.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_100_03 = sample_from_mfcc(read_mfcc('student_samples/100/01/100-021.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_100_04 = sample_from_mfcc(read_mfcc('student_samples/100/01/100-032.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_100_05 = sample_from_mfcc(read_mfcc('student_samples/100/01/100-044.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_100_06 = sample_from_mfcc(read_mfcc('student_samples/100/01/100-008.wav', SAMPLE_RATE), NUM_FRAMES)
   
    mfcc_101_01 = sample_from_mfcc(read_mfcc('student_samples/101/01/101-001.wav', SAMPLE_RATE), NUM_FRAMES)    
    mfcc_101_02 = sample_from_mfcc(read_mfcc('student_samples/101/01/101-029.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_101_03 = sample_from_mfcc(read_mfcc('student_samples/101/01/101-023.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_101_04 = sample_from_mfcc(read_mfcc('student_samples/101/01/101-016.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_101_05 = sample_from_mfcc(read_mfcc('student_samples/101/01/101-013.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_101_06 = sample_from_mfcc(read_mfcc('student_samples/101/01/101-008.wav', SAMPLE_RATE), NUM_FRAMES)

    mfcc_102_01 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-001.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_102_02 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-013.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_102_03 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-017.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_102_04 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-004.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_102_05 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-006.wav', SAMPLE_RATE), NUM_FRAMES)
    mfcc_102_06 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-012.wav', SAMPLE_RATE), NUM_FRAMES)
    
    #mfcc_102_01 = sample_from_mfcc(read_mfcc('student_samples/102/01/102-001.wav', SAMPLE_RATE), NUM_FRAMES)
    sum_100 = 0
    sum_100 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_100_01, axis=0)), predict_utt)
    sum_100 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_100_02, axis=0)), predict_utt)
    sum_100 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_100_03, axis=0)), predict_utt)
    sum_100 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_100_04, axis=0)), predict_utt)
    sum_100 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_100_05, axis=0)), predict_utt)
    sum_100 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_100_06, axis=0)), predict_utt)

    sum_101 = 0
    sum_101 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_101_01, axis=0)), predict_utt)
    sum_101 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_101_02, axis=0)), predict_utt)
    sum_101 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_101_03, axis=0)), predict_utt)
    sum_101 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_101_04, axis=0)), predict_utt)
    sum_101 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_101_05, axis=0)), predict_utt)
    sum_101 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_101_06, axis=0)), predict_utt)

    sum_102 = 0
    sum_102 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_102_01, axis=0)), predict_utt)
    sum_102 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_102_02, axis=0)), predict_utt)
    sum_102 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_102_03, axis=0)), predict_utt)
    sum_102 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_102_04, axis=0)), predict_utt)
    sum_102 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_102_05, axis=0)), predict_utt)
    sum_102 +=  batch_cosine_similarity(model.m.predict(np.expand_dims(mfcc_102_06, axis=0)), predict_utt)

    speaker_id = 'none'
    max_cosine_sum = 0
    if sum_100 >= max_cosine_sum and sum_100 > 3:
        speaker_id = '100'
        max_cosine_sum = sum_100
    if sum_101 >= max_cosine_sum and sum_101 > 3:
        speaker_id = '101'
        max_cosine_sum = sum_101
    if sum_102 >= max_cosine_sum and sum_102 > 3:
        speaker_id = '102'
        max_cosine_sum = sum_102
    return speaker_id, max_cosine_sum / 6

print("Enter filename >>", end=' ')
filename = input()
res = predict_id('ResCNN_checkpoint_36.h5', filename)
print(res)
