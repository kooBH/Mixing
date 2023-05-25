"""
noise ensegment script.
2023.05.25, KooBH@github.com

"""


dir_in =  "/home/data2/kbh/UMA8_data/noise"
dir_out = "/home/data2/kbh/UMA8_data/noise_seg"

sr = 16000
unit_seg = sr*60

#########

import os,glob
import numpy
import librosa as rs
import soundfile as sf

# utils
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')


list_in = glob.glob(os.path.join(dir_in,"**","*.wav"),recursive=True)

def enseg(idx):
    path  = list_in[idx]

    name = path.split("/")[-1]
    id = name.split(".")[0]

    x,_ = rs.load(path,sr=sr,mono=False)

    idx_sample = 0
    len_x = x.shape[1]
    while idx_sample < len_x : 
        seg = x[:,idx_sample:idx_sample+unit_seg]
        name_out = "{}_{}.wav".format(id,idx_sample)        

        sf.write(os.path.join(dir_out,name_out),seg.T,sr)

        idx_sample += unit_seg

if __name__ == "__main__" : 
    os.makedirs(os.path.join(dir_out),exist_ok=True)

    cpu_num = int(cpu_count()/2)
    arr = list(range(len(list_in)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(enseg, arr), total=len(arr),ascii=True,desc='enseg'))