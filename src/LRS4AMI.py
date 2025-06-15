"""
Mixing LRS dataset to fit to AMI dataset

AMI mic array

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
from moviepy.editor import *
from modules.align import align

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
n_ch = hp.audio.n_channel

# Load List
list_video = []
for path in hp.data.video:
    list_video += glob.glob(os.path.join(path,"**","*.mp4"),recursive=True)
print("video : {}".format(len(list_video)))

os.makedirs(dir_out + "/noise",exist_ok=True)
os.makedirs(dir_out + "/noisy",exist_ok=True)
os.makedirs(dir_out + "/clean",exist_ok=True)

## Common parameters for RIR generation
nb_rcv = 8
fs = 16000

# AMI-array shape
geo_mic = np.array(
[[+0.1000, +0.0000, +0.0000],
 [+0.0707, +0.0707, +0.0000],
 [+0.0000, +0.1000, +0.0000],
 [-0.0707, +0.0707, +0.0000],
 [-0.1000, +0.0000, +0.0000],
 [-0.0707, -0.0707, +0.0000],
 [-0.0000, -0.1000, +0.0000],
 [+0.0707, -0.0707, +0.0000]
])
mic_pattern = "omni"
orV_rcv = None # None for omni

def mix(idx):
    import gpuRIR
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(True)

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    # parsing

    path = np.random.choice(list_video)
    LRS = path.split("/")[-4]
    dir_LRS = path.split("/")[-3]
    id_dir = path.split("/")[-2]
    file_name = path.split("/")[-1]
    file_name = file_name.split(".")[0]

    # extract audio
    video = VideoFileClip(path)
    audio = video.audio
    sr = audio.fps
    # Extract the audio as a list of samples
    audio_samples = list(audio.iter_frames())
    # Convert the list of samples to a NumPy array
    raw = np.array(audio_samples)[:,0]

    # resample to target sr
    raw = rs.resample(raw,orig_sr = sr,target_sr=hp.audio.sr)

    sz_x = np.random.uniform(hp.RIR.sz_x[0],hp.RIR.sz_x[1])
    sz_y = np.random.uniform(hp.RIR.sz_y[0],hp.RIR.sz_y[1])
    sz_z = np.random.uniform(hp.RIR.sz_z[0],hp.RIR.sz_z[1])
    room_sz = np.array([sz_x,sz_y,sz_z])

    angle = np.random.randint(0,360)
    dist = np.random.randint(hp.RIR.dist[0],hp.RIR.dist[1])
    rad = np.deg2rad(angle)

    ps_x = dist*np.sin(rad) + room_sz[0]/2
    ps_y = dist*np.cos(rad) + room_sz[1]/2
    ps_z = np.random.uniform(1,2)
    pos_src = np.array([ps_x,ps_y,ps_z])


    pos_rcv = np.array([pos_src[0]/2, pos_src[0]/2, 1] + geo_mic)


    #abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
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

    output = []
    for i in range(nb_rcv) : 
        output.append(np.convolve(raw,RIRs[0,i,:]))
    output = np.array(output)

    raw = np.expand_dims(raw,0)
    raw,tau = align(raw,output)
    if raw is None :
        print("WARNING::Failed to align sample, skip this item : {}".format(idx))
        return
    output = output[:,:raw.shape[1]]
    output = np.array(output)

    # Adjust Scale
    output = output/np.max(np.abs(output))
    raw = raw/np.max(np.abs(raw))
   
    # Save
    name_out = "{}_{}_{}_{}_{}_{}.wav".format(idx,angle,LRS,dir_LRS,id_dir,file_name)
    name_clean = "{}_clean_{}_{}_{}_{}.wav".format(idx,LRS,dir_LRS,id_dir,file_name)
    path_wav_out = os.path.join(dir_out,"noisy",name_out)
    path_wav_clean = os.path.join(dir_out,"clean",name_clean)

    sf.write(path_wav_out,output.T,hp.audio.sr)
    sf.write(path_wav_clean,raw.T,hp.audio.sr)

if __name__ == "__main__" : 
    cpu_num = int(cpu_count()/4)
   # cpu_num = 16

    arr = list(range(args.n_data))

    for i in tqdm(range(args.n_data)) : 
        mix(i)

#    with Pool(cpu_num) as p:
#        r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc=str("processing : {}".format(args.config))))

