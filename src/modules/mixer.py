import numpy as np

import sys,os
sys.path.append("../gpuRIR")

def sample_mic_pos(pool,shuffle_channel = False):
    # sample
    idx =  np.random.randint(0,len(pool))
    mic_pos = pool[idx]

    if shuffle_channel :
        np.random.shuffle(mic_pos)

    return mic_pos

def sample_space(pool_room, 
                 dist_min=0.5,
                 height_min=1.0,
                 height_max=2.5,
                 n_src = 2,
                 angle_resolution = 15
                 ):
    # sample
    idx =  np.random.randint(0,len(pool_room))
    room = np.array(pool_room[idx])
    center = room/2
    center[2] = height_min

    # TODO : random noise on mic pos

    pt_mic = center

    dist_max = min(center[:2])-dist_min

    pt_src = []
    pool_angle = np.arange(0,360,angle_resolution)
    angle = np.random.choice(pool_angle, n_src,replace=False)
    
    for i in range(n_src):
        t_angle = angle[i]
        theta = np.deg2rad(t_angle)

        d = np.random.rand()*dist_max + dist_min
        mx = np.cos(theta)*d + center[0]
        my = np.sin(theta)*d + center[1]
        phi = np.deg2rad(15 + np.random.rand()*15)
        mz = np.tan(phi)*d + height_min

        # clip
        #mx = np.clip(mx,dist_min, room[0]-dist_min)
        #my = np.clip(my,dist_min, room[1]-dist_min)
        mz = np.clip(mz,height_min,height_max)

        pt_src.append(np.array([mx,my,mz]))
    
    return room,pt_mic,pt_src,angle

def cal_RIR(T60,mic,room_sz,pt_mic,pt_src,fs=16000):
    # https://github.com/DavidDiazGuerra/gpuRIR/issues/13
    # Note that the gpuRIR should be imported inside func otherwise you get an "Incorrect Initialization" Error.
    import gpuRIR   
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(True)

    nb_rcv = len(mic) # Number of receivers
    pos_rcv = mic + pt_mic

    mic_pattern = "omni" # Receiver polar pattern
    orV_rcv=None # None for omni

    abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls

    att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
    att_max = 60.0 # Attenuation at the end of the simulation [dB]

    beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

    #print("room\n{}\n src\n{}\n rcv\n{}".format(room_sz,pos_src,pos_rcv))

    RIRs = []
    for i in range(len(pt_src)):
        RIR = gpuRIR.simulateRIR(room_sz, beta, pt_src[i], pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
        RIRs.append(RIR[0])
    return RIRs


def gen_RIR(rgn_RT60, mic_pos, room,pt_mic,pt_src):
    RT60 = np.random.uniform(rgn_RT60[0],rgn_RT60[1])
    RIRs = cal_RIR(RT60,mic_pos,room,pt_mic,pt_src)
    return RIRs, RT60

"""
    x : [L] for 1D signal, [L, C] for 2D signal
"""
def sample_audio(x,n_sample) :
    L = x.shape[0]
    
    if L <= n_sample : 
        shortage = n_sample - L
        x = np.pad(x,(0,shortage))
        return x, -1
    else :
        left = L - n_sample
        start = np.random.randint(0,left)
        x = x[start:start+n_sample]
        return x, start
"""
Modify the length of audio signal to n_sample
with random start point and length

args 
    x : [ L ]
    n_sample : total length of audio signal
    min_sample : minimum length of audio signal

return 
    raw : [ n_sample ]
    start_sample : start point of input in return
    idx_sample : start point of input in original signal
    len_sample : length of input in original signal



"""    
def distribute_audio(x,n_sample) : 
    raw = np.zeros(n_sample)
    
    len_x = x.shape[0]
    # len_x < n_sample
    if len_x < n_sample :
        shortage = n_sample - len_x
        start = np.random.randint(0,shortage)
        raw[start:start+len_x] = x
        return raw, start, 0, len_x
    # len_x >= n_sample
    else :
        start = np.random.randint(0,len_x-n_sample)
        raw = x[start:start+n_sample]
        return raw, 0, start, n_sample

def manage_SNR() :
    pass


def mix(audio ) :
    pass



if __name__ == "__main__" :
    print("test generation")
    pass