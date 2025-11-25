import os
import torch
from transformers import Trainer, TrainingArguments
from safetensors.torch import save_file

# Internal Modules
from src.config import TrainConfig
from src.dataset import ChatterboxDataset, data_collator
from src.model import resize_and_load_t3_weights, ChatterboxTrainerWrapper
from src.utils import setup_logger

# Chatterbox Imports
from src.chatterbox_.tts import ChatterboxTTS
from src.chatterbox_.models.t3.t3 import T3


logger = setup_logger("ChatterboxFinetune")


def main():
    
    cfg = TrainConfig()
    
    logger.info("--- Starting Chatterbox Finetuning ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Device: {device}")
    logger.info(f"Model Directory: {cfg.model_dir}")

    # 1. LOAD ORIGINAL MODEL TEMPORARILY
    logger.info("Loading original model to extract weights...")
    # Loading on CPU first to save VRAM
    tts_engine_original = ChatterboxTTS.from_local(cfg.model_dir, device="cpu")

    pretrained_t3_state_dict = tts_engine_original.t3.state_dict()
    original_t3_config = tts_engine_original.t3.hp

    # 2. CREATE NEW T3 MODEL WITH NEW VOCAB SIZE
    logger.info(f"Creating new T3 model with vocab size: {cfg.new_vocab_size}")
    
    new_t3_config = original_t3_config
    new_t3_config.text_tokens_dict_size = cfg.new_vocab_size

    new_t3_model = T3(hp=new_t3_config)

    # 3. TRANSFER WEIGHTS
    logger.info("Transferring weights...")
    new_t3_model = resize_and_load_t3_weights(new_t3_model, pretrained_t3_state_dict)

    # Clean up memory
    del tts_engine_original
    del pretrained_t3_state_dict

    # 4. PREPARE ENGINE FOR TRAINING
    # Reload engine components (VoiceEncoder, S3Gen) but inject our new T3
    tts_engine_new = ChatterboxTTS.from_local(cfg.model_dir, device="cpu")
    tts_engine_new.t3 = new_t3_model 

    # Freeze other components
    logger.info("Freezing S3Gen and VoiceEncoder...")
    for param in tts_engine_new.ve.parameters(): 
        param.requires_grad = False
        
    for param in tts_engine_new.s3gen.parameters(): 
        param.requires_grad = False

    # Enable Training for T3
    tts_engine_new.t3.train()
    for param in tts_engine_new.t3.parameters(): param.requires_grad = True

    # 5. DATASET & WRAPPER
    logger.info("Initializing Dataset...")
    train_ds = ChatterboxDataset(cfg, tts_engine_new)
    
    model_wrapper = ChatterboxTrainerWrapper(tts_engine_new.t3)

    # 6. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_epochs,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        remove_unused_columns=False, # Required for our custom wrapper
        dataloader_num_workers=0,    
        report_to=["tensorboard"],
        fp16=True if torch.cuda.is_available() else False,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model_wrapper,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator
    )

    logger.info("Starting Training Loop...")
    trainer.train()

    # 7. SAVE FINAL MODEL
    logger.info("Training complete. Saving model...")
    os.makedirs(cfg.output_dir, exist_ok=True)
    final_model_path = os.path.join(cfg.output_dir, "t3_finetuned.safetensors")

    # Save only the T3 weights
    save_file(tts_engine_new.t3.state_dict(), final_model_path)
    logger.info(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()