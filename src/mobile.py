"""
Mobile mixing
mic : top, bottom = 2ch
clean : KspnSpeech_eval
noise : CHiME4

last update : 2025.02.24

1. Clean for 2ch smartphone mic
2. Diffuse noise at 4 corners
3. Mixing
"""
import sys,os
sys.path.append("gpuRIR")

from utils.hparams import HParam

from modules.align import *
from modules.mixer import *
from glob import glob

import numpy as np
import librosa as rs
import soundfile as sf
import torch

import pandas as pd
import random

# utils
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True,
                    help="yaml for configuration")
parser.add_argument('--default', type=str, required=True,
                    help="default configuration")
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--version_name', '-v', type=str, required=True,
                    help="version of current training")
args = parser.parse_args()

hp = HParam(args.config,args.default)

dir_out = args.output
dir_clean = hp.data.clean
dir_noise = hp.data.noise
n_audio = hp.data.n_output
sec = hp.mix.sec
fs = hp.audio.sr
nb_rcv = hp.audio.n_channels

eps = 1e-13
os.makedirs(dir_out+"/clean",exist_ok=True)
os.makedirs(dir_out+"/deverb",exist_ok=True)
os.makedirs(dir_out+"/noisy",exist_ok=True)
#os.makedirs(dir_out+"/tmp",exist_ok=True)
"""
Mic geometric references

Galaxy S23
146.3 mm (5.76 in) H
70.8 mm (2.79 in) W
7.6 mm (0.30 in) D

Galaxy S23 Ultra: 
163.2 mm (6.43 in) H
77.9 mm (3.07 in) W
8.9 mm (0.35 in) D

Galaxy A33
159.7 mm (6.29 in) H
74.0 mm (2.9 in) W
8.1 mm (0.32 in) D

iPhone 16
H : 147.6mm
W : 71.6mm
D : 7.80mm

iPhone 16+
H : 160.9mm
W : 77.8mm
D : 7.80mm
"""
geo_min = np.array([
    # bottom
    [0.01, 0.0, 0.0], # 1cm,0,0
    # top
    [0.01, 0.14, 0.0 ]  # 1cm, 14cm, 0
    ]
    )
geo_max = np.array([
    # bottom
    [0.06, 0.0, 0.0], # 6cm,0,0
    # top
    [0.06, 0.17,0.0 ]  # 6cm, 17cm, 0
    ]
    )

mic_pattern = "omni"
orV_rcv = None # None for omni

list_speech = []
if type(dir_clean) is list : 
    for d in dir_clean : 
        list_speech += [x for x in glob(os.path.join(d,"**","*.wav"),recursive=True)]
        list_speech += [x for x in glob(os.path.join(d,"**","*.flac"),recursive=True)]
else :
    list_speech += [x for x in glob(os.path.join(dir_clean,"**","*.wav"),recursive=True)]
    list_speech += [x for x in glob(os.path.join(dir_clean,"**","*.flac"),recursive=True)]
random.shuffle(list_speech)

print("speech : {}".format(len(list_speech)))

list_noise = []
list_noise += [x for x in glob(os.path.join(dir_noise,"*.wav"))]

print("noise : {}".format(len(list_noise)))

def mix(idx):
    import gpuRIR
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(True)

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    #### Clean ####

    # parsing
    path = list_speech[idx]
    file_name = path.split("/")[-1]
    file_name = file_name.split(".")[0]

    # Load
    raw = rs.load(path,sr=hp.audio.sr)[0]

    # Adjust Length
    if raw.shape[0] < hp.audio.sr*sec :
        start_pad = np.random.randint(0,hp.audio.sr*sec-raw.shape[0])
        end_pad = hp.audio.sr*sec-raw.shape[0]-start_pad
        raw = np.pad(raw,(start_pad,end_pad),'constant')
    elif raw.shape[0] > hp.audio.sr*sec :
        start_idx = np.random.randint(0,raw.shape[0]-hp.audio.sr*sec)
        raw = raw[start_idx:start_idx+hp.audio.sr*sec]

    # Generate Room size
    sz_x = np.random.uniform(hp.RIR.sz_x[0],hp.RIR.sz_x[1])
    sz_y = np.random.uniform(hp.RIR.sz_y[0],hp.RIR.sz_y[1])
    sz_z = np.random.uniform(hp.RIR.sz_z[0],hp.RIR.sz_z[1])
    room_sz = np.array([sz_x,sz_y,sz_z])

    # Generate Target Position
    angle = np.random.randint(0,360)
    dist = np.random.uniform(hp.RIR.dist[0],hp.RIR.dist[1])
    rad = np.deg2rad(angle)

    ps_x = dist*np.sin(rad) + room_sz[0]/2
    ps_y = dist*np.cos(rad) + room_sz[1]/2
    ps_z = np.random.uniform(1,2)
    pos_src = np.array([ps_x,ps_y,ps_z])

    # Set Receiver Position
    geo_mic = np.random.uniform(geo_min,geo_max)
    pos_rcv = np.array([pos_src[0]/2, pos_src[0]/2, 1] + geo_mic)

    #abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
    # Generate Room Configuration
    T60 = np.random.uniform(hp.RIR.RT[0], hp.RIR.RT[1]) # Time for the RIR to reach 60dB of attenuation [s]
    att_diff = np.random.uniform(hp.RIR.att_diff[0],hp.RIR.att_diff[1])	# Attenuation when start using the diffuse reverberation model [dB]
    att_max = np.random.uniform(hp.RIR.att_max[0],hp.RIR.att_max[1]) # Attenuation at the end of the simulation [dB]

    #beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
    beta = gpuRIR.beta_SabineEstimation(room_sz, T60) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

    try : 
        RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
    except AssertionError : 
        print("ERROR::simulateRIR() failed\n room_sz {}\npos_src {}\n pos_rcv {}".format(room_sz,pos_src,pos_rcv))
        return

    speech = []
    for i in range(nb_rcv) : 
        speech.append(np.convolve(raw,RIRs[0,i,:]))
    speech = np.array(speech)

    #sf.write(dir_out+"/tmp/raw.wav",raw,hp.audio.sr)
    #sf.write(dir_out+"/tmp/speech.wav",speech.T,hp.audio.sr)

    raw = np.expand_dims(raw,0)
    raw2,tau = align(raw,speech)

    if tau > int(0.3 * raw.shape[-1]) : 
        pass
    else : 
        raw = raw2

    if raw is None :
        print("WARNING::Failed to align sample, skip this item : {}".format(idx))
        return
    speech = speech[:,:raw.shape[1]]
    speech = np.array(speech)

    #sf.write(dir_out+"/tmp/raw_algin.wav",raw.T,hp.audio.sr)

    # Adjust Scale
    speech = speech/np.max(np.abs(speech))
    raw = raw/np.max(np.abs(raw))


    #sf.write(dir_out+"/tmp/speech_norm.wav",speech.T,hp.audio.sr)
    #sf.write(dir_out+"/tmp/raw_norm.wav",raw.T,hp.audio.sr)

    ### Noise ####
    if hp.data.noise == "white" :
        noise_raw = np.random.randn(hp.audio.sr*sec)
    else : 
        noise_raw = rs.load(np.random.choice(list_noise),sr=hp.audio.sr)[0]
        # Adjust Length
        if noise_raw.shape[0] < hp.audio.sr*sec :
            noise_raw = np.pad(noise_raw,(0,hp.audio.sr*sec-noise_raw.shape[0]),'constant')
        elif noise_raw.shape[0] > hp.audio.sr*sec :
            start_idx = np.random.randint(0,noise_raw.shape[0]-hp.audio.sr*sec)
            noise_raw = noise_raw[start_idx:start_idx+hp.audio.sr*sec]

    noise = None
    #sf.write(dir_out+"/tmp/noise.wav",noise_raw,hp.audio.sr)

    T60_noise = np.random.uniform(hp.RIR.RT_noise[0], hp.RIR.RT_noise[1])
    beta = gpuRIR.beta_SabineEstimation(room_sz, T60_noise)
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60_noise)
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60_noise)
    nb_img = gpuRIR.t2n( Tdiff, room_sz )

    ## For each corners ##
    for i in range(4) : 
        room_sz = np.array([sz_x,sz_y,sz_z])
        pos_src = room_sz.copy()
        # 30cm from ceiling
        pos_src[2] -= 0.3
        # North West
        if i == 0 :
            pos_src[0] = 0.3
            pos_src[1] -= 0.3
        # North East
        elif i == 1 :
            pos_src[0] -= 0.3
            pos_src[1] -= 0.3
        # South West
        elif i == 2 :
            pos_src[0] = 0.3
            pos_src[1] = 0.3
        # South East
        elif i == 3 :
            pos_src[0] -= 0.3
            pos_src[1] = 0.3

        try : 
            RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
        except AssertionError : 
            print("ERROR::simulateRIR() failed\n room_sz {}\npos_src {}\n pos_rcv {}".format(room_sz,pos_src,pos_rcv))
            return
        t_noise = []
        for i in range(nb_rcv) : 
            t_noise.append(np.convolve(noise_raw,RIRs[0,i,:]))
        t_noise = np.array(t_noise)
        if noise is None : 
            noise = t_noise
        else : 
            noise += t_noise

    # Adjust Length
    if noise.shape[1] > hp.audio.sr*sec :
        start_idx = np.random.randint(0,noise.shape[1]-hp.audio.sr*sec)
        noise = noise[:,start_idx:start_idx+hp.audio.sr*sec]

    #sf.write(dir_out+"/tmp/noise_RIR.wav",noise.T,hp.audio.sr)

    # Adjust Scale
    noise = noise/np.max(np.abs(noise))
    #sf.write(dir_out+"/tmp/noise_norm.wav",noise.T,hp.audio.sr)

    ### Mixing ###

    # SNR
    SNR = np.random.uniform(hp.mix.SNR[0],hp.mix.SNR[1])

    speech_rms = (speech[0] ** 2).sum() ** 0.5
    noise_rms = (noise[0] ** 2).sum() ** 0.5

    snr_scalar = speech_rms / (10 ** (SNR / 10)) / (noise_rms + eps)
    noise *= snr_scalar
    #sf.write(dir_out+"/tmp/noise_SNR.wav",noise.T,hp.audio.sr)
    noisy = speech + noise
    #sf.write(dir_out+"/tmp/noisy.wav",noisy.T,hp.audio.sr)

    # Adjsut Scale
    target_dB_FS = hp.mix.target_dB_FS + np.random.uniform(-hp.mix.target_dB_FS_floating,hp.mix.target_dB_FS_floating)

    denom = np.max(np.abs(noisy))
    noisy /=denom
    speech /=denom
    raw /=denom
    #sf.write(dir_out+"/tmp/noisy_norm.wav",noisy.T,hp.audio.sr)
    #sf.write(dir_out+"/tmp/speech_denom.wav",speech.T,hp.audio.sr)
    #sf.write(dir_out+"/tmp/raw_denom.wav",noise.T,hp.audio.sr)
    
    rms = np.sqrt(np.mean(noisy[0] ** 2))
    scalar = 10 ** (target_dB_FS / 10) / (rms + eps)
    noisy *= scalar
    speech *= scalar
    raw *= scalar

    #sf.write(dir_out+"/tmp/noisy_final.wav",noisy.T,hp.audio.sr)
    #sf.write(dir_out+"/tmp/speech_final.wav",speech.T,hp.audio.sr)
    #sf.write(dir_out+"/tmp/raw_final.wav",noise.T,hp.audio.sr)
   
    # Save
    T60_str = "{:.1f}".format(T60)
    T60_str = T60_str.replace(".","p")

    dist_str = "{:.1f}".format(dist)
    dist_str = dist_str.replace(".","p")

    name_out = "idx_{}_angle_{}_dist_{}_T60_{}.wav".format(idx,angle,dist_str,T60_str)
    name_speech = "idx_{}_{}.wav".format(idx,file_name)
    path_wav_out = os.path.join(dir_out,"noisy",name_out)
    path_wav_raw = os.path.join(dir_out,"deverb",name_speech)
    path_wav_clean= os.path.join(dir_out,"clean",name_speech)

    sf.write(path_wav_out,noisy.T,hp.audio.sr)
    sf.write(path_wav_raw,raw.T,hp.audio.sr)
    sf.write(path_wav_clean,speech.T,hp.audio.sr)

if __name__ == "__main__" : 
    cpu_num = int(cpu_count()/4)
   # cpu_num = 16

    arr = list(range(n_audio))

    for i in tqdm(range(n_audio)) : 
        mix(i)

#    with Pool(cpu_num) as p:
#        r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc=str("processing : {}".format(args.config))))


