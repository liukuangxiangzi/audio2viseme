

#python audio_feature_extractor.py -i audio/data/path/ -d number-of-delay-frames -c number-of-context-frames -o output/data/path
#e.g. python audio_feature_extractor.py -i data/test_audio/ -d 1 -c 5 -o data/result/

import librosa
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import os, shutil
from tqdm import tqdm
import utils
import argparse


def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="input speech file")
parser.add_argument("-d", "--delay", type=int, help="Delay in terms of number of frames, where each frame is 40 ms")
parser.add_argument("-c", "--ctx", type=int, help="context window size")
parser.add_argument("-o", "--out_fold", type=str, help="output folder")
args = parser.parse_args()

output_path = args.out_fold
num_features_Y = 136
num_frames = 75
wsize = 0.04
hsize = wsize
fs = 48000#44100
trainDelay = args.delay
ctxWin = args.ctx

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)


audio_folder_path = args.in_file
generated_all = []
cur_features_to_save = []

for root, dirnames, filenames in os.walk(audio_folder_path):
    filenames.sort()
    if os.path.exists(root + '.DS_Store') is True:
        filenames.remove('.DS_Store')
    for filename in filenames:
        # Used for padding zeros to first and second temporal differences
        zeroVecD = np.zeros((1, 64), dtype='f16')
        zeroVecDD = np.zeros((2, 64), dtype='f16')

        # Load speech and extract features
        sound, sr = librosa.load(audio_folder_path + filename, sr=fs) #(132300,) sr=50000
        print("sound,sr", np.shape(sound), sr)

        melFrames = np.transpose(utils.melSpectra(sound, sr, wsize, hsize))
        print("melFrames", np.shape(melFrames))
        melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
        print("melDelta", np.shape(melDelta))
        melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
        print("melDDelta", np.shape(melDDelta))

        features = np.concatenate((melDelta, melDDelta), axis=1) #76,128
        print("features1", np.shape(features))
        features = addContext(features, ctxWin)  #76,768
        print("features2 contex5", np.shape(features))
        features = np.reshape(features, (1, features.shape[0], features.shape[1]))  #1,76,768
        print("features3", np.shape(features))


        upper_limit = features.shape[1]
        print("upper_limit", upper_limit)
        lower = 0
        generated = np.zeros((0, num_features_Y))

        # Generates face landmarks one-by-one
        # This part can be modified to predict the whole sequence at one, but may introduce discontinuities
        for i in tqdm(range(upper_limit)):
            cur_features = np.zeros((1, num_frames, features.shape[2]))
            if i+1 > 75:
                lower = i+1-75
            cur_features[:,-i-1:,:] = features[:,lower:i+1,:]
        cur_features.resize(np.shape(cur_features)[1],np.shape(cur_features)[2])
        print(np.shape(cur_features))
        cur_features_to_save.append(cur_features)
np.save(output_path + '/Xtext.npy', cur_features_to_save)