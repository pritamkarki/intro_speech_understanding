import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    # Frame parameters
    frame_length = int(0.025 * Fs)  # 25ms
    step = int(0.01 * Fs)  # 10ms
    
    # Calculate energy in each frame
    num_frames = (len(waveform) - frame_length) // step + 1
    energies = []
    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length]
        energy = np.sum(frame ** 2)
        energies.append(energy)
    
    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)
    
    # Find segments where energy > threshold
    segments = []
    in_segment = False
    segment_start = 0
    
    for i, energy in enumerate(energies):
        if energy > threshold and not in_segment:
            in_segment = True
            segment_start = i * step
        elif energy <= threshold and in_segment:
            in_segment = False
            segment_end = i * step
            segments.append(waveform[segment_start:segment_end])
    
    # Handle case where segment extends to end
    if in_segment:
        segment_end = num_frames * step
        segments.append(waveform[segment_start:segment_end])
    
    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    
    for segment in segments:
        # Pre-emphasis
        pre_emphasized = np.zeros_like(segment)
        pre_emphasized[0] = segment[0]
        pre_emphasized[1:] = segment[1:] - 0.97 * segment[:-1]
        
        # Frame parameters for spectrogram
        frame_length = int(0.004 * Fs)  # 4ms
        step = int(0.002 * Fs)  # 2ms
        
        # Calculate number of frames
        num_frames = (len(pre_emphasized) - frame_length) // step + 1
        
        # Calculate STFT
        mstft = np.zeros((num_frames, frame_length))
        for i in range(num_frames):
            start = i * step
            frame = pre_emphasized[start:start + frame_length]
            mstft[i] = np.abs(np.fft.fft(frame))
        
        # Convert to spectrogram (log magnitude)
        floor = 0.001 * np.amax(mstft)
        mstft_floor = np.maximum(floor, mstft)
        spectrogram = 20 * np.log10(mstft_floor)
        
        # Keep only low-frequency half
        low_freq_spectrogram = spectrogram[:, :frame_length // 2]
        
        # Average across frames to create model
        model = np.mean(low_freq_spectrogram, axis=0)
        models.append(model)
    
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    # VAD to get test segments
    test_segments = VAD(testspeech, Fs)
    
    # Convert test segments to models
    test_models = segments_to_models(test_segments, Fs)
    
    # Calculate cosine similarity between each test model and each training model
    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    
    for i, model in enumerate(models):
        for j, test_model in enumerate(test_models):
            # Cosine similarity
            dot_product = np.dot(model, test_model)
            norm_model = np.linalg.norm(model)
            norm_test = np.linalg.norm(test_model)
            if norm_model > 0 and norm_test > 0:
                sims[i, j] = dot_product / (norm_model * norm_test)
    
    # Find best match for each test segment
    test_outputs = []
    for j in range(K):
        best_model_idx = np.argmax(sims[:, j])
        test_outputs.append(labels[best_model_idx])
    
    return sims, test_outputs
