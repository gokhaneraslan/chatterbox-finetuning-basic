import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger



logger = setup_logger(__name__)

def preprocess_dataset_ljspeech(config, tts_engine: ChatterboxTTS):
    
    data = pd.read_csv(config.csv_path, sep="|", header=None, quoting=3)
    
    os.makedirs(config.preprocessed_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)
    
    logger(f"Processing dataset... Total: {len(data)}")

    success_count = 0

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        
        try:
            
            filename = str(row[0])
            if not filename.endswith(".wav"): 
                filename += ".wav"
            
            wav_path = os.path.join(config.wav_dir, filename)
            
            if not os.path.exists(wav_path): 
                continue


            wav, sr = torchaudio.load(wav_path)
            
            if wav.shape[0] > 1: 
                wav = wav.mean(dim=0, keepdim=True)
                
            if sr != S3_SR:
                resampler = torchaudio.transforms.Resample(sr, S3_SR)
                wav = resampler(wav)
            
            wav = wav.to(device)


            with torch.no_grad():

                wav_np = wav.cpu().squeeze().numpy()
                
                spk_emb_np = tts_engine.ve.embeds_from_wavs([wav_np], sample_rate=S3_SR)
                speaker_emb = torch.from_numpy(spk_emb_np[0]).cpu()


                s_tokens, _ = tts_engine.s3gen.tokenizer(wav.unsqueeze(0))
                speech_tokens = s_tokens.squeeze().cpu()


                prompt_samples = int(config.prompt_duration * S3_SR)
                
                if wav.shape[1] < prompt_samples:
                    prompt_wav = torch.nn.functional.pad(wav, (0, prompt_samples - wav.shape[1]))
                    
                else:
                    prompt_wav = wav[:, :prompt_samples]
                
                p_tokens, _ = tts_engine.s3gen.tokenizer(prompt_wav.unsqueeze(0))
                prompt_tokens = p_tokens.squeeze().cpu()


            raw_text = str(row[2]) if len(row) > 2 else str(row[1])
            
            clean_text = punc_norm(raw_text)
            
            text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()


            save_path = os.path.join(config.preprocessed_dir, filename.replace(".wav", ".pt"))
            
            torch.save({
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens
            }, save_path)
            
            success_count += 1
        
        except Exception as e:
            logger.error(f"Error ({filename}): {e}")
            continue
        
    logger.info(f"Preprocessing completed! Success: {success_count}/{len(data)}")