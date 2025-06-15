import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
def compute_shift(x, y):
    assert x.shape[-1] == y.shape[-1]
    c = cross_correlation_using_fft(x, y)
    assert c.shape[-1] == x.shape[-1]
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift

def align(target,ref):
    assert target.ndim == 2, "target.ndim != 2"
    assert ref.ndim == 2, "ref.ndim != 2"

    if target.shape[-1] < ref.shape[-1] : 
        ref = ref[:,:target.shape[-1]]
    elif target.shape[-1] > ref.shape[-1] :
        target = target[:,:ref.shape[-1]]

    tau = compute_shift(target[0,:],ref[0,:])


   # print("{} {} | {}".format(target.shape,ref.shape,tau))

    n_sample = target.shape[-1]

    data_synced  = np.zeros_like(target)
    if tau > 0 : 
        data_synced[:,tau:] = target[:,:n_sample-tau]
    else :
        data_synced[:,:n_sample+tau] = target[:,-tau:n_sample]
        
    return data_synced, tau