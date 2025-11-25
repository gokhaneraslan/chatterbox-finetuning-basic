import os
import torch
import numpy as np
import soundfile as sf
import random
import re
from safetensors.torch import load_file

from src.utils import setup_logger, trim_silence_with_vad
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("ChatterboxInference")



BASE_MODEL_DIR = os.getenv("MODEL_DIR", "./pretrained_models")
FINETUNED_WEIGHTS = os.getenv("FINETUNED_WEIGHTS", "./chatterbox_output/t3_finetuned.safetensors")
NEW_VOCAB_SIZE = 2454 # Must match the training vocab size


def load_finetuned_engine(device):
    """
    Loads the Chatterbox engine and replaces the T3 module with the fine-tuned version.
    """
    
    logger.info(f"Loading base model from: {BASE_MODEL_DIR}")

    # 1. Load Base Engine (CPU first)
    tts_engine = ChatterboxTTS.from_local(BASE_MODEL_DIR, device="cpu")
    
    # 2. Configure New T3 Model
    logger.info(f"Initializing new T3 with vocab size: {NEW_VOCAB_SIZE}")
    
    t3_config = tts_engine.t3.hp
    t3_config.text_tokens_dict_size = NEW_VOCAB_SIZE
    
    # Create fresh T3 instance
    new_t3 = T3(hp=t3_config)
    
    # 3. Load Fine-Tuned Weights
    if os.path.exists(FINETUNED_WEIGHTS):
        
        logger.info(f"Loading fine-tuned weights: {FINETUNED_WEIGHTS}")
        
        state_dict = load_file(FINETUNED_WEIGHTS)
        
        # Load weights (strict=False enables loading even if some metadata differs)
        try:
            
            new_t3.load_state_dict(state_dict, strict=False)
            logger.info("Fine-tuned weights loaded successfully.")
            
        except RuntimeError as e:
            logger.error(f"Weight mismatch: {e}")
            raise e
        
    else:
        
        logger.warning(f"Fine-tuned file not found at {FINETUNED_WEIGHTS}. Using random init (Garbage output expected).")


    # 4. Swap and Move to Device
    tts_engine.t3 = new_t3
    tts_engine.t3.eval()
    tts_engine.to(device)
    
    return tts_engine


def generate_sentence_audio(
        engine: ChatterboxTTS,
        text: str,
        prompt_path: str,
        lang_id: str = "tr",
        **kwargs
    ):
    """
    Generates audio for a single sentence and trims silence.
    """
    
    try:
        
        # Generate raw wav [1, T]
        # Note: ChatterboxTTS.generate expects specific kwargs
        wav_tensor = engine.generate(
            text=text,
            audio_prompt_path=prompt_path,
            language_id=lang_id, # If supported by your modified generate
            **kwargs
        )
        
        # Convert to numpy for VAD
        wav_np = wav_tensor.squeeze().cpu().numpy()
        
        # Trim Noise/Silence
        trimmed_wav = trim_silence_with_vad(wav_np, engine.sr)
        
        return engine.sr, trimmed_wav
        
    except Exception as e:
        logger.error(f"Error generating sentence '{text[:20]}...': {e}")
        return 24000, np.zeros(0)


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Inference running on: {device}")

    # --- INPUT PARAMETERS ---
    TEXT_TO_SAY = " "
    
    AUDIO_PROMPT = "/content/drive/MyDrive/audio/MyTTSDataset/wavs/sample1.wav"
    OUTPUT_FILE = "output_stitched.wav"
    
    # Hyperparameters
    PARAMS = {
        "temperature": 0.8,
        "exaggeration": 1.2, # Controls expressiveness
        "cfg_weight": 0.3,   # Classifier Free Guidance scale
        "repetition_penalty": 2.0 
    }
    
    # 1. Load Model
    engine = load_finetuned_engine(device)
    
    # 2. Split Text (Simple Logic)
    # Splits by . ? ! but keeps the punctuation
    sentences = re.split(r'(?<=[.?!])\s+', TEXT_TO_SAY.strip())
    sentences = [s for s in sentences if s.strip()]
    
    logger.info(f"Found {len(sentences)} sentences.")
    
    all_chunks = []
    sample_rate = 24000
    
    # 3. Inference Loop
    set_seed(42) # Fixed seed for consistency
    
    for i, sent in enumerate(sentences):
        
        logger.info(f"Synthesizing ({i+1}/{len(sentences)}): {sent}")
        
        sr, audio_chunk = generate_sentence_audio(
            engine, 
            sent, 
            AUDIO_PROMPT, 
            **PARAMS
        )
        
        if len(audio_chunk) > 0:
            all_chunks.append(audio_chunk)
            sample_rate = sr
            
            # Add short pause between sentences (e.g., 200ms)
            pause_samples = int(sr * 0.2)
            all_chunks.append(np.zeros(pause_samples, dtype=np.float32))


    # 4. Concatenate and Save
    if all_chunks:
        
        final_audio = np.concatenate(all_chunks)
        sf.write(OUTPUT_FILE, final_audio, sample_rate)
        
        logger.info(f"Result saved to: {OUTPUT_FILE}")
        
    else:
        logger.error("No audio generated.")


if __name__ == "__main__":
    main()