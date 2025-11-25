import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.utils import setup_logger


logger = setup_logger(__name__)



class ChatterboxDataset(Dataset):
    
    def __init__(self, config):
        self.cfg = config
        self.preprocessed_dir = config.preprocessed_dir
        
        # List only files with .pt extension
        if not os.path.exists(self.preprocessed_dir):
            raise FileNotFoundError(f"Preprocessing folder not found: {self.preprocessed_dir}.")
            
        self.files = [f for f in os.listdir(self.preprocessed_dir) if f.endswith(".pt")]
        
        if len(self.files) == 0:
            raise RuntimeError(f"There are no .pt files in the folder: {self.preprocessed_dir}")
            
        logger.info(f"Dataset loaded. Total sample: {len(self.files)}")

        self.sot_token = config.start_text_token 
        self.eot_token = config.stop_text_token


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        try:
            
            filename = self.files[idx]
            
            pt_path = os.path.join(self.preprocessed_dir, filename)
            
            data = torch.load(pt_path)
            
            
            text_tokens = data["text_tokens"]
            if text_tokens.size(0) > self.cfg.max_text_len - 2:
                text_tokens = text_tokens[:self.cfg.max_text_len - 2]
                
            sot = torch.tensor([self.sot_token], dtype=torch.long)
            eot = torch.tensor([self.eot_token], dtype=torch.long)
            text_tokens = torch.cat([sot, text_tokens, eot])

            # 2. Speech Tokens
            speech_tokens = data["speech_tokens"]
            if speech_tokens.size(0) > self.cfg.max_speech_len:
                speech_tokens = speech_tokens[:self.cfg.max_speech_len]

            return {
                "text_tokens": text_tokens,
                "speech_tokens": speech_tokens,
                "speaker_emb": data["speaker_emb"],
                "prompt_tokens": data["prompt_tokens"]
            }


        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
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


    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "prompt_tokens": prompt_tokens
    }