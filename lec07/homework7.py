import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    # Calculate frequencies for major chord
    # Root: f
    # Major third: 4 semitones above f = f * 2^(4/12)
    # Major fifth: 7 semitones above f = f * 2^(7/12)
    f_root = f
    f_third = f * 2**(4/12)
    f_fifth = f * 2**(7/12)
    
    # Generate 0.5 seconds of audio
    N = int(0.5 * Fs)
    n = np.arange(N)
    
    # Generate each tone
    omega_root = 2 * np.pi * f_root / Fs
    omega_third = 2 * np.pi * f_third / Fs
    omega_fifth = 2 * np.pi * f_fifth / Fs
    
    x_root = np.cos(omega_root * n)
    x_third = np.cos(omega_third * n)
    x_fifth = np.cos(omega_fifth * n)
    
    # Combine the three tones
    x = x_root + x_third + x_fifth
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)
    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    # Compute the FFT
    X = np.fft.fft(x)
    
    # Get the magnitude spectrum
    magnitude = np.abs(X)
    
    # Get the frequency axis (only positive frequencies)
    N = len(x)
    freqs = np.fft.fftfreq(N, 1/Fs)
    
    # Only consider positive frequencies
    positive_freqs = freqs[:N//2]
    positive_magnitude = magnitude[:N//2]
    
    # Find the indices of the three largest magnitudes
    top3_indices = np.argsort(positive_magnitude)[-3:]
    
    # Get the corresponding frequencies
    top3_freqs = positive_freqs[top3_indices]
    
    # Sort in ascending order
    top3_freqs = np.sort(top3_freqs)
    
    return top3_freqs[0], top3_freqs[1], top3_freqs[2]
