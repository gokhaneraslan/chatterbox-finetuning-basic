import logging
import sys
import torch
import torchaudio
import numpy as np



def setup_logger(name: str, level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger


_VAD_MODEL = None
_GET_SPEECH_TIMESTAMPS = None

def load_vad_model():
    """Lazy loads the Silero VAD model."""
    
    global _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    if _VAD_MODEL is not None:
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    try:
        
        print("Loading Silero VAD model...")
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        _GET_SPEECH_TIMESTAMPS = utils[0]
        _VAD_MODEL = model
        
        print("Silero VAD loaded.")
        
        return _VAD_MODEL, _GET_SPEECH_TIMESTAMPS
    
    except Exception as e:
        print(f"Error loading VAD: {e}")
        return None, None


def trim_silence_with_vad(audio_waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Trims silence/noise from the end of the audio using Silero VAD.
    """
    
    vad_model, get_timestamps = load_vad_model()
    if vad_model is None:
        return audio_waveform

    VAD_SR = 16000
    # Convert numpy to tensor
    audio_tensor = torch.from_numpy(audio_waveform).float()

    # Resample for VAD if necessary
    if sample_rate != VAD_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=VAD_SR)
        vad_input = resampler(audio_tensor)
        
    else:
        vad_input = audio_tensor

    try:
        # Get speech timestamps
        speech_timestamps = get_timestamps(vad_input, vad_model, sampling_rate=VAD_SR)
        
        if not speech_timestamps:
            return audio_waveform

        # Get the end of the last speech chunk
        last_speech_end_vad = speech_timestamps[-1]['end']

        # Scale back to original sample rate
        scale_factor = sample_rate / VAD_SR
        cut_point = int(last_speech_end_vad * scale_factor)

        trimmed_wav = audio_waveform[:cut_point]
        
        return trimmed_wav


    except Exception as e:
        print(f"VAD trimming failed: {e}")
        return audio_waveform