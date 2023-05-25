import os,glob
import argparse
import scipy
import numpy as np
import librosa as rs
import soundfile as sf
from scipy import io
import random
import json

# utils
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

from utils.hparams import HParam
from modules.mixer import sample_audio
from modules.align import align

# param
parser = argparse.ArgumentParser()
parser.add_argument('--output_root', '-o', type=str, required=True)
parser.add_argument('--config', '-c', type=str, required=True)
parser.add_argument('--default', '-d', type=str, required=True)
parser.add_argument('--n_data', '-n', type=int, required=True)

args = parser.parse_args()
hp = HParam(args.config,args.default)

## ROOT
output_root = args.output_root
n_data = args.n_data

list_speech = []
n_sample = int(hp.mix.sec * hp.audio.sr)
n_src = hp.mix.n_src
max_SIR = hp.mix.max_SIR
min_scale_dB = hp.mix.scale_dB
sr = hp.audio.sr
n_ch = hp.audio.n_channel

for path in hp.data.speech:
    list_speech += glob.glob(os.path.join(path,"**","*.wav"),recursive=True)
print("speech : {}".format(len(list_speech)))

list_RIR = glob.glob(os.path.join(hp.RIR.root,"*.mat"),recursive=True)
list_RIR.sort()
print("RIR : {}".format(len(list_RIR)))
#print(list_RIR)

list_noise = glob.glob(os.path.join(hp.data.noise,"*.wav"),recursive=True)
print("noise : {}".format(len(list_noise)))

def process(idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    cur_src = np.random.randint(1,n_src+1)
    
    ## RIR
    # select angles
    idx_angles = np.random.choice(len(list_RIR),cur_src,replace=False)

    # Mixing 
    signals = []
    raws = []
    angles = []
    #Speeches
    for i in range(cur_src) : 
        # (n_RIR,n_ch) == (65536,7)
        path = list_RIR[idx_angles[i]]
        name = path.split("/")[-1]
        deg = int(name.split("_")[2])
        angles.append(deg)

        RIR = io.loadmat(path)["ir"]

        # TODO : select cross part of UMA-8
        RIR = RIR[:,[2,3,5,6]]

        idx_src = np.random.randint(0,len(list_speech))
        raw = rs.load(list_speech[idx_src],sr=hp.audio.sr)[0]
        raw,idx_start = sample_audio(raw,n_sample)

        # RIR
        signal = []
        for j in range(n_ch) : 
            s = scipy.signal.convolve(raw,RIR[:,j])
            s = s[:n_sample]
            signal.append(s)
        signal = np.array(signal)

        raw = np.expand_dims(raw,0)
        raw = align(raw,signal)

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

    # Mix Speech
    x = np.sum(signals,axis=0)

    ## Noise Mix
    if(hp.mix.use_noise) : 
        idx_noise = np.random.randint(0,len(list_noise))

        # SNR
        speech_rms = np.sqrt(np.mean(x ** 2))
        snr_scaler = ref_rms / (10 ** (SIR/20)) / (sig_rms + 1e-13)
        noise *= snr_scaler

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
    label["angles"] = angles
    label["n_src"] = cur_src
    # 3,4,6,7 channels of UMA8
    label["mic_pos"] = [
        [0.0372, 0.0215, 0],
        [ 0.0372,-0.0215, 0],
        [-0.0372,-0.0215, 0],
        [-0.0372,0.0215, 0],
    ]
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
