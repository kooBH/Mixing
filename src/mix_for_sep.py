import os, glob
import argparse

import numpy as np
import librosa as rs
import soundfile as sf
import scipy.signal

import json

# utils
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

from utils.hparams import HParam
from modules.mixer import sample_mic_pos,sample_space,gen_RIR,sample_audio
from modules.align import align

# param
parser = argparse.ArgumentParser()
parser.add_argument('--output_root', '-o', type=str, required=True)
parser.add_argument('--config', '-c', type=str, required=True)
parser.add_argument('--default', '-d', type=str, required=True)
parser.add_argument('--mic', '-m', type=str, required=True)
parser.add_argument('--room', '-r', type=str, required=True)
parser.add_argument('--n_data', '-n', type=int, required=True)

args = parser.parse_args()
hp = HParam(args.config,args.default)
hp_mic = HParam(args.mic)
hp_room= HParam(args.room)

## ROOT
output_root = args.output_root

n_data = args.n_data

list_speech = []
n_sample = int(hp.mix.sec * hp.audio.sr)
n_src = hp.mix.n_src
max_SIR = hp.mix.max_SIR
min_scale_dB = hp.mix.scale_dB
sr = hp.audio.sr

for path in hp.data.speech:
    list_speech += glob.glob(os.path.join(path,"**","*.wav"),recursive=True)
print("speech : {}".format(len(list_speech)))

def process(idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    # RIR
    mic_pos = sample_mic_pos(hp_mic["array"])
    room,pt_mic, pt_srcs,angles = sample_space(hp_room["sz"],n_src = n_src,angle_resolution=hp.RIR.angle_resolution)
    RIRs, RT60 = gen_RIR(hp.RIR.rgn_RT60, mic_pos, room,pt_mic,pt_srcs)
    #print("angle : {} | RT60 : {:.2f} | RIR : {}".format(angles,RT60,RIRs))

    # Mixing 
    signals = []
    raws = []
    #Speeches
    for i in range(n_src) : 
        idx_src = np.random.randint(0,len(list_speech))
        raw = rs.load(list_speech[idx_src],sr=hp.audio.sr)[0]
        raw,idx_start = sample_audio(raw,n_sample)

        # RIR
        raw = np.expand_dims(raw,0)
        signal = scipy.signal.convolve(raw,RIRs[i])
        signal = signal[:,:n_sample]

        raw = align(raw,signal)

        if raw is None :
            print("ERROR: align failed, Skip this index : {}".format(idx))
            return

        SIR = np.random.uniform(0,max_SIR)

        # Scaling : first src as reference
        if i != 0 :
            sig_rms = np.sqrt(np.mean(signal**2))
            snr_scaler = ref_rms / (10 ** (SIR/20)) / (sig_rms + 1e-13)
            signal *= snr_scaler
            raw *= snr_scaler
        else : 
            ref_rms = np.sqrt(np.mean(signal**2))

        signals.append(signal)
        raws.append(raw)
    signals = np.array(signals)
    raws = np.array(raws)

    # Mix
    x = np.sum(signals,axis=0)

    # dB adjustment
    scale_dB = np.random.uniform(0,min_scale_dB)
    rms = np.sqrt(np.mean(x ** 2))
    scalar = 10 ** (scale_dB / 20) / (rms + 1e-13)
    x *= scalar
    signals*= scalar
    raws *= scalar

    # clip
    if np.any(np.abs(x) > 0.999)  : 
        noisy_scalar = np.max(np.abs(x)) / (0.99 - 1e-13)  # same as divide by 1
        x /= noisy_scalar
        signals /= noisy_scalar

    if np.any(np.abs(raws) > 0.999)  : 
        noisy_scalar = np.max(np.abs(raws)) / (0.99 - 1e-13)  # same as divide by 1
        raws /= noisy_scalar

    # Label
    label = {}
    label["room"] = room.tolist()
    label["mic_pos"] = mic_pos
    label["pt_mic"] = pt_mic.tolist()
    label["pt_srcs"] = np.array(pt_srcs).tolist()
    label["angles"] = angles.tolist()
    label["SIR"] = SIR
    label["scale_dB"] = scale_dB

    # Save
    path_label = os.path.join(output_root,"label","{}.json".format(idx))
    with open(path_label,'w') as f:
        json.dump(label,f,indent=4)

    path_noisy= os.path.join(output_root,"noisy","{}.wav".format(idx))
    sf.write(path_noisy,x.T,sr)

    for i in range(n_src) : 
        path_clean = os.path.join(output_root,"clean","{}_{}.wav".format(idx,i))
        if hp.mix.unreverbed_clean : 
            sf.write(path_clean,raws[i,0].T,sr)
        else :
            sf.write(path_clean,signals[i,0].T,sr)

if __name__=='__main__': 
    cpu_num = int(cpu_count()/2)

    for subdir in ["label","noisy","clean"] : 
        os.makedirs(os.path.join(output_root,subdir),exist_ok=True)

    arr = list(range(n_data))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc=str("processing : {}".format(args.config))))

