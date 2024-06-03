"""
LRS Mixing with RIRed audio for separation
"""

import os,glob
import argparse
import numpy as np
import librosa as rs
import soundfile as sf
from utils.hparams import HParam
import json
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Parse arg
parser = argparse.ArgumentParser()
parser.add_argument('--dir_out', '-o', type=str, required=True)
parser.add_argument('--config', '-c', type=str, required=True)
parser.add_argument('--default', '-d', type=str, required=True)
parser.add_argument('--n_data', '-n', type=int, required=True)
args = parser.parse_args()
hp = HParam(args.config,args.default)

# Params
dir_out = args.dir_out
n_data = args.n_data
n_sample = int(hp.mix.sec * hp.audio.sr)
n_frame =  int(hp.mix.sec * hp.video.fps)
min_src = hp.mix.min_src
max_src = hp.mix.max_src
max_SIR = hp.mix.max_SIR
ratio_src = hp.mix.ratio_src
scale_dB = hp.mix.scale_dB
sr = hp.audio.sr
n_ch = hp.audio.n_channel
ratio_av = hp.video.fps/hp.audio.sr
max_start = hp.mix.max_start

# Load List
list_audio = glob.glob(os.path.join(hp.data.audio,"*"))
print(len(list_audio))

os.makedirs(dir_out,exist_ok=True)

def mix(idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    if len(ratio_src) == max_src : 
        cur_src = int(np.random.choice(max_src,1,ratio_src)[0]+1)
    else :
        cur_src = np.random.randint(min_src,max_src+1)
    """
    + start_clean_audio/video
        : starting index of extractecd sample in original audio/video
    + length_clean_audio/video
        : length of extractecd sample in original audio/video
    + start_noisy_audio/video
        : starting index of extractecd sample in mixed audio/video
    + length_noisy_audio/video
        : length of extractecd sample in mixed audio/video
    """
    id_person = []
    id_spk = []
    filename = []

    start_clean_audio = []
    length_clean_audio = []
    start_noisy_audio = []
    length_noisy_audio = []

    start_clean_video = []
    length_clean_video = []
    start_noisy_video = []
    length_noisy_video = []
    SIR = []
    angle = []
    n_sample = hp.mix.sec*hp.audio.sr
    n_frame = hp.mix.sec*hp.video.fps

    #print("cur_src " , cur_src)
    audio_plate = np.zeros((hp.audio.n_channel,n_sample))

    # Sample data
    for i in range(cur_src) : 
        audio = np.random.choice(list_audio,1)[0]
        tokens = audio.split("/")[-1]
        person = tokens.split("_")[0]
        spk = tokens.split("_")[1]
        list_candiate = glob.glob(os.path.join(audio,"*m.wav"))
        # e.g. xFSCX9mAHPo_00003_R907_340_1.5m.wav
        candiate = np.random.choice(list_candiate,1)[0]
        tokens = candiate.split("/")[-1]
        filename.append(tokens)
        cur_angle = tokens.split("_")[3]

        # No duplicated angle 
        for j in range(len(id_person)) : 
            while angle[j] == cur_angle : 
                candiate = np.random.choice(list_candiate,1)[0]
                tokens = candiate.split("/")[-1]
                cur_angle = tokens.split("_")[3]

        # sample audio 
        audio = rs.load(candiate,sr=sr,mono=False)[0]
        L_i = audio.shape[1]
        L_i_orig = L_i

        if L_i > n_sample : 
            SCA = np.random.randint(0,L_i - n_sample)
            L_i = n_sample
        else : 
            SCA = 0

        SNA = np.random.randint(0, max_start * n_sample)
        if SNA + L_i > n_sample :
            L_i = n_sample - SNA

        utterance = audio[:,SCA:SCA+L_i]

        # Distribute Audio with SIR
        if i != 0 :
            tSIR = np.random.randint(0,max_SIR)
            sig_rms = np.sqrt(np.mean(utterance[0]**2))
            snr_scaler = ref_rms / (10 ** (tSIR/20)) / (sig_rms + 1e-13)
            utterance *= snr_scaler
        else : 
            t_scale_dB = np.random.uniform(scale_dB[0],scale_dB[1])
            ref_rms = np.sqrt(np.mean(utterance[0]**2))
            scalar = 10 ** (t_scale_dB / 20) / (ref_rms + 1e-13)
            utterance *= scalar
            ref_rms = np.sqrt(np.mean(utterance[0]**2))
            tSIR = 0
            #print("{} {} {} ".format(idx, t_scale_dB,scalar))
        audio_plate[:,SNA:SNA+L_i] += utterance
        # Match Video
        SCV = int(SCA * ratio_av)
        SNV = int(SNA * ratio_av)
        VL_i = int(L_i * ratio_av)

        # label 
        id_person.append(person)
        id_spk.append(spk)
        angle.append(cur_angle)
        SIR.append(tSIR)

        start_clean_audio.append(SCA)
        length_clean_audio.append(L_i)
        start_noisy_audio.append(SNA)
        length_noisy_audio.append(L_i)

        start_clean_video.append(SCV)
        length_clean_video.append(VL_i)
        start_noisy_video.append(SNV)
        length_noisy_video.append(VL_i)

    # Adjust Scale
    if np.abs(audio_plate).any() > 1 :
        audio_plate = audio_plate /np.max(np.abs(audio_plate))
   

    # Generate Label
    label = {}
    label["id_person"] = id_person
    label["id_spk"] = id_spk
    label["start_clean_audio"] = start_clean_audio
    label["length_clean_audio"] = length_clean_audio
    label["start_noisy_audio"] = start_noisy_audio
    label["length_noisy_audio"] = length_noisy_audio
    label["start_clean_video"] = start_clean_video
    label["length_clean_video"] = length_clean_video
    label["start_noisy_video"] = start_noisy_video
    label["length_noisy_video"] = length_noisy_video
    label["SIR"] = SIR
    label["n_sample"] = n_sample
    label["n_frame"] = n_frame
    label["scale_dB"] = t_scale_dB
    label["filename"] = filename

    # Save
    path_wav_out = os.path.join(dir_out,"{}.wav".format(idx))
    path_label_out = os.path.join(dir_out,"{}.json".format(idx))

    sf.write(path_wav_out,audio_plate.T,sr)
    with open(path_label_out,'w') as f : 
        json.dump(label,f,indent=4)

if __name__ == "__main__" : 
    cpu_num = int(cpu_count()/2)

    arr = list(range(args.n_data))

#    for i in range(10) : 
#        mix(i)

    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc=str("processing : {}".format(args.config))))


