import os,glob
import argparse
import numpy as np
import librosa as rs
import soundfile as sf
import random


# utils
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

# param
parser = argparse.ArgumentParser()
parser.add_argument('--dir_clean', '-i', type=str, required=True)
parser.add_argument('--dir_noise', '-n', type=str, required=True)
parser.add_argument('--dir_out', '-o', type=str, required=True)
parser.add_argument('--SNR', '-s', type=int, required=True)
parser.add_argument('--sr', type=int, default=16000)

args = parser.parse_args()

SNR = args.SNR
sr = args.sr

dir_clean = args.dir_clean
dir_noise = args.dir_noise
dir_out = args.dir_out

## listing

list_clean = glob.glob(os.path.join(dir_clean,"**","*.flac"),recursive=True)
list_noise = glob.glob(os.path.join(dir_noise,"**","*.wav"),recursive=True)
print("noise : {}".format(len(list_noise)))

def process(idx):
    sr = 16000
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    clean, sr = rs.load(list_clean[idx],sr=sr)
    noise, sr = rs.load(random.choice(list_noise),sr=sr)
    len_noise = len(noise)

    # align length of audio
    while len_noise < len(clean) : 
        app, sr = rs.load(random.choice(list_noise),sr=sr)
        noise = np.concatenate((noise,app))
        len_noise += len(app)
    noise = noise[:len(clean)] 

    clean_rms = np.sqrt(np.mean(clean**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    scale = clean_rms / (noise_rms * 10**(SNR/20))    
    noise = noise * scale

    noisy = clean + noise
    noisy = noisy / np.max(np.abs(noisy))

    name_file = os.path.basename(list_clean[idx]).split(".")[0]
    sf.write(os.path.join(dir_out,name_file+".wav"),noisy,sr)

if __name__=='__main__': 
    cpu_num = int(cpu_count()/2)

    os.makedirs(os.path.join(dir_out),exist_ok=True)

    arr = list(range(len(list_clean)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc=str("processing")))
