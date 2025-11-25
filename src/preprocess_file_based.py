import os
import glob
import torch
import torchaudio
from tqdm import tqdm

from src.chatterbox_.tts import ChatterboxTTS, punc_norm
from src.chatterbox_.models.s3tokenizer import S3_SR
from src.utils import setup_logger



logger = setup_logger(__name__)



def preprocess_dataset_file_based(config, tts_engine: ChatterboxTTS):
    """
    Reads .wav and .txt file pairs in a folder, processes them, and saves them as .pt.
    Structure:

    ID.wav (Audio)

    ID.txt (Text)

    """

    os.makedirs(config.preprocessed_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts_engine.ve.to(device)
    tts_engine.s3gen.to(device)
    tts_engine.ve.eval()
    tts_engine.s3gen.eval()

    search_path = os.path.join(config.wav_dir, "*.wav")
    wav_files = glob.glob(search_path)
    
    if len(wav_files) == 0:
        logger.error(f"ERROR: No .wav files found in folder '{config.wav_dir}'!")
        return

    logger.info(f"Processing dataset... Found audio file: {len(wav_files)}")

    success_count = 0
    
    for wav_path in tqdm(wav_files, desc="Preprocessing"):
        try:
            filename = os.path.basename(wav_path)
            file_id = os.path.splitext(filename)[0]
            

            txt_path = os.path.join(config.wav_dir, f"{file_id}.txt")
            
            if not os.path.exists(txt_path):
                logger.warning(f"Text file not found, skipping: {file_id}")
                continue
                
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()
                
            if not raw_text:
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


            clean_text = punc_norm(raw_text)
            # Tokenizer
            text_tokens = tts_engine.tokenizer.text_to_tokens(clean_text).squeeze(0).cpu()

            # --- 5. SAVING ---
            # We keep the file name: ID.pt
            save_path = os.path.join(config.preprocessed_dir, f"{file_id}.pt")
            
            torch.save({
                "speech_tokens": speech_tokens,
                "speaker_emb": speaker_emb,
                "prompt_tokens": prompt_tokens,
                "text_tokens": text_tokens,
            }, save_path)
            
            success_count += 1

        except Exception as e:
            logger.error(f"Error ({filename}): {e}")
            continue

    logger.info(f"Preprocessing completed! Success: {success_count}/{len(wav_files)}")
