"""
Apply RIRs on each audio sample
"""
import os,glob
import argparse
import scipy
import numpy as np
import librosa as rs
import soundfile as sf
from scipy import io
import random
import json
from moviepy.editor import *
import cv2

# utils
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

from modules.align import align

# param
parser = argparse.ArgumentParser()
parser.add_argument('--dir_LRS', '-i', type=str, required=True)
parser.add_argument('--dir_rir', '-r', type=str, required=True)
parser.add_argument('--dir_out', '-o', type=str, required=True)
args = parser.parse_args()

# Load list of videos
list_clip = glob.glob(os.path.join(args.dir_LRS,"**","*.mp4"),recursive=True)

# Load list of rir
list_RIR = glob.glob(os.path.join(args.dir_rir,"*.mat"),recursive=True)

# Apply RIR on every clean audio


def apply_RIR(idx):
    path_video = list_clip[idx]

    tokens = path_video.split("/")
    person = tokens[-2]
    name = tokens[-1].split(".")[0]
    spk = name.split(".")[0]

    video = VideoFileClip(path_video)
    audio = video.audio
    # Extract the audio as a list of samples
    audio_samples = list(audio.iter_frames())
    # Convert the list of samples to a NumPy array
    raw = np.array(audio_samples)[:,0]

    sr = audio.fps
    raw = rs.resample(raw,orig_sr = sr,target_sr=16000)

    dir_out = os.path.join(args.dir_out,person + "_" + spk)
    os.makedirs(dir_out,exist_ok=True)

    # Apply RIR
    for i_rir in range(len(list_RIR)) : 
        # TSP_deg_090_room_R907_dist_1.5m_20230524.mat
        path_RIR = list_RIR[i_rir]
        name_RIR = path_RIR.split("/")[-1]
        tokens = name_RIR.split("_")
        angle = tokens[2]
        dist = tokens[6]
        room = tokens[4]

        RIR = io.loadmat(path_RIR)["ir"]
        # NOTE : select cross part of UMA-8
        RIR = RIR[:,[2,3,5,6]]

        # RIR
        signal = []
        for j in range(4) : 
            s = scipy.signal.convolve(raw,RIR[:,j])
            signal.append(s)
        signal = np.array(signal)

        # algin
        signal, tau = align(signal,np.expand_dims(raw,0))
        
        # Save
        name_out = "{}_{}_{}_{}_{}.wav".format(person,spk,room,angle,dist)
        path_out = os.path.join(dir_out,name_out)
        sf.write(path_out,signal.T,16000)
    name_out = "{}_{}.wav".format(person,spk)
    path_out = os.path.join(dir_out,name_out)
    sf.write(path_out,raw,16000)

if __name__=='__main__': 
    cpu_num = int(cpu_count()/2)

    arr = list(range(len(list_clip)))

    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(apply_RIR, arr), total=len(arr),ascii=True,desc=str("processing : {}".format(args.dir_out))))

    #for i in tqdm(range(len(arr))) : 
    #   process(i)
