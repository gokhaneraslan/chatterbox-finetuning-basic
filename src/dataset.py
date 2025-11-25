import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger


logger = setup_logger(__name__)



class ChatterboxDataset(Dataset):
    
    def __init__(self, config, tts_engine: ChatterboxTTS):
        
        self.cfg = config
        # Load CSV (Format: File|Text|NormText or similar)
        # quoting=3 prevents errors with quotes in text
        self.data = pd.read_csv(config.csv_path, sep="|", header=None, quoting=3)

        # Extract components from the engine
        self.text_tokenizer = tts_engine.tokenizer
        self.speech_tokenizer = tts_engine.s3gen.tokenizer
        self.voice_encoder = tts_engine.ve
        self.t3_config = tts_engine.t3.hp

        # Samples for prompt (3 sec * 16000 = 48000)
        self.prompt_samples = int(config.prompt_duration * S3_SR)
        

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        
        filename = "Unknown"
        
        try:
            
            row = self.data.iloc[idx]
            filename = str(row[0])
            
            if not filename.endswith(".wav"): 
                filename += ".wav"

            # Use 3rd column if available (Normalized), else 2nd
            raw_text = str(row[2]) if len(row) > 2 else str(row[1])

            # 1. AUDIO PROCESSING
            wav_path = os.path.join(self.cfg.wav_dir, filename)
            
            if not os.path.exists(wav_path):
                logger.warning(f"File not found: {wav_path}")
                return None

            # Load audio at native sampling rate first
            wav, sr = torchaudio.load(wav_path)

            # Convert Stereo to Mono
            if wav.shape[0] > 1: 
                wav = wav.mean(dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sr != S3_SR:
                resampler = torchaudio.transforms.Resample(sr, S3_SR)
                wav = resampler(wav) # [1, T]

            # 2. PREPARE COMPONENTS

            # A) Speaker Embedding
            wav_numpy = wav.squeeze().numpy()
            with torch.no_grad():
                # embeds_from_wavs expects a list of arrays
                spk_emb_np = self.voice_encoder.embeds_from_wavs([wav_numpy], sample_rate=S3_SR)
                speaker_emb = torch.from_numpy(spk_emb_np[0]) # [EmbDim]

            # B) Prompt Tokens (Reference Audio)
            # Pad if shorter than prompt duration
            if wav.shape[1] < self.prompt_samples:
                prompt_wav = F.pad(wav, (0, self.prompt_samples - wav.shape[1]))
                
            else:
                prompt_wav = wav[:, :self.prompt_samples]


            with torch.no_grad():
                p_tokens, _ = self.speech_tokenizer(prompt_wav)
                prompt_tokens = p_tokens.squeeze(0) # [PromptLen]


            # C) Target Speech Tokens (Full Audio)
            with torch.no_grad():
                s_tokens, _ = self.speech_tokenizer(wav)
                speech_tokens = s_tokens.squeeze(0)
                
                # Truncate length
                if speech_tokens.size(0) > self.cfg.max_speech_len:
                    speech_tokens = speech_tokens[:self.cfg.max_speech_len]

            # D) Text Tokens
            clean_text = punc_norm(raw_text)
            text_tokens = self.text_tokenizer.text_to_tokens(clean_text).squeeze(0)

            # Add Start/Stop tokens for T3
            sot = torch.tensor([self.t3_config.start_text_token], dtype=torch.long)
            eot = torch.tensor([self.t3_config.stop_text_token], dtype=torch.long)
            
            text_tokens = torch.cat([sot, text_tokens, eot])

            if text_tokens.size(0) > self.cfg.max_text_len:
                text_tokens = text_tokens[:self.cfg.max_text_len]
                

            return {
                "text_tokens": text_tokens.cpu(),
                "speech_tokens": speech_tokens.cpu(),
                "speaker_emb": speaker_emb.float().cpu(),
                "prompt_tokens": prompt_tokens.cpu()
            }


        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return None



def data_collator(batch):

    batch = [item for item in batch if item is not None]
    if not batch: 
        return {}

    # Padding
    text_tokens = pad_sequence([x["text_tokens"] for x in batch], batch_first=True, padding_value=0)
    speech_tokens = pad_sequence([x["speech_tokens"] for x in batch], batch_first=True, padding_value=0)
    prompt_tokens = pad_sequence([x["prompt_tokens"] for x in batch], batch_first=True, padding_value=0)

    speaker_embs = torch.stack([x["speaker_emb"] for x in batch])

    # Lengths (Required for masking)
    text_lens = torch.tensor([len(x["text_tokens"]) for x in batch], dtype=torch.long)
    speech_lens = torch.tensor([len(x["speech_tokens"]) for x in batch], dtype=torch.long)

    # Create Labels (for Loss Calculation)
    # Mask padding with -100
    labels_speech = speech_tokens.clone()
    for i, length in enumerate(speech_lens):
        labels_speech[i, length:] = -100 


    labels_text = text_tokens.clone()
    for i, length in enumerate(text_lens):
        labels_text[i, length:] = -100


    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens,
        "labels_text": labels_text,
        "labels_speech": labels_speech
    }