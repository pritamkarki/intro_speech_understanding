import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    nframes = int((len(speech) - frame_length) / frame_skip)
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for k in range(nframes):
        start = k * frame_skip
        frame = speech[start:start + frame_length]
        
        X = np.zeros((frame_length - order, order))
        for j in range(order):
            X[:, j] = frame[order - 1 - j:frame_length - 1 - j]
        
        target = frame[order:]
        a, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        
        A[k, 0] = 1
        A[k, 1:] = a
        
        e = np.copy(frame)
        for n in range(order, frame_length):
            for j in range(1, order + 1):
                e[n] += a[j - 1] * frame[n - j]
        excitation[k, :] = e
    
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = A.shape[0]
    order = A.shape[1] - 1
    frame_length = len(e) // nframes
    synthesis = np.zeros(len(e))
    
    for k in range(nframes):
        start = k * frame_skip
        for n in range(frame_length):
            idx = start + n
            y_n = e[idx]
            for j in range(1, min(n + 1, order + 1)):
                y_n -= A[k, j] * synthesis[idx - j]
            synthesis[idx] = y_n
    
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]
    total_length = frame_skip * nframes
    
    gain = np.sqrt(np.mean(excitation ** 2, axis=1))
    
    e_robot = np.zeros(total_length)
    t = np.arange(total_length)
    base = np.zeros(total_length)
    base[::T0] = -1
    
    for k in range(nframes):
        start = k * frame_skip
        end = start + frame_skip
        e_robot[start:end] = base[start:end] * gain[k]
    
    return gain, e_robot
