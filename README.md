# Chatterbox TTS: Fine-Tuning & Inference Kit 🎙️

A modular infrastructure for **fine-tuning** the Chatterbox TTS model (specifically the T3 module) with your own dataset and generating high-quality speech synthesis.

Specially designed to support **new languages** (like Turkish) that aren't fully supported by the original model by building a custom tokenizer structure and expanding the model's vocabulary.

---

## ⚠️ CRITICAL INFORMATION (Please Read)

### 0. Preprocessing is Mandatory
This repository uses an **offline preprocessing** strategy to maximize training speed. This script processes all audio files, extracts speaker embeddings and acoustic tokens, and saves them as `.pt` files.

### 1. Tokenizer and Vocab Size (Most Important)
Chatterbox uses a grapheme-based (character-level) tokenizer. The `tokenizer.json` file downloaded by `setup.py` includes support for **23 languages** from the original Chatterbox repository, covering most common characters across multiple languages.

*   **Default Support:** The provided tokenizer already includes characters for English, Turkish, French, German, Spanish, and 18+ other languages
*   **When to customize:** If your target language has special characters not covered in the default tokenizer, you can create a custom `tokenizer.json`
*   **Examples of special characters by language:**
    *   Turkish: `ç, ğ, ş, ö, ü, ı`
    *   French: `é, è, ê, à, ù, ç`
    *   German: `ä, ö, ü, ß`
    *   Spanish: `ñ, á, é, í, ó, ú`
*   **Critical:** The `NEW_VOCAB_SIZE` variable in both `src/config.py` AND `inference.py` **must exactly match** the total number of tokens in your `tokenizer.json` file
*   **Default vocab size:** Check the downloaded `tokenizer.json` to see the exact token count, then set `NEW_VOCAB_SIZE` accordingly

### 2. Audio Sample Rates
*   **Training (Input):** Chatterbox's encoder and T3 module work with **16,000 Hz (16kHz)** audio. Even if your dataset uses different rates, `dataset.py` automatically resamples to 16kHz.
*   **Output (Inference):** The model's vocoder generates audio at **24,000 Hz (24kHz)**.

---

## 📂 Folder Structure

```text
chatterbox-finetune/
├── pretrained_models/                             # setup.py downloads required models here
│   ├── ve.safetensors
│   ├── s3gen.safetensors
│   ├── t3.safetensors
│   └── tokenizer.json
├── MyTTSDataset/                                  # Your custom dataset in LJSpeech format
│   ├── metadata.csv                               # Dataset metadata (file|text|normalized_text)
│   └── wavs/                                      # Directory containing WAV files
├── FileBasedDataset/                              # Your custom dataset in LJSpeech format
│   ├── 0a0bc5d3-f195-464a-8716-d6e01fd4784f.txt   # Dataset metadata (text)
│   └── 0a0bc5d3-f195-464a-8716-d6e01fd4784f.wav   # WAV files
├── speaker_reference/                             # Speaker reference audio files
│   └── reference.wav                              # Reference audio for voice cloning
├── src/
│   ├── config.py                                  # All settings and hyperparameters
│   ├── dataset.py                                 # Data loading and processing
│   ├── model.py                                   # Model weight transfer and training wrapper
|   ├── preprocess_ljspeech.py                     # Preprocessing script
|   ├── preprocess_file_based.py                   # Preprocessing script
│   └── utils.py                                   # Logger and VAD utilities
├── train.py                                       # Main training script
├── inference.py                                   # Speech synthesis script (with VAD support)
├── setup.py                                       # Setup script for downloading models
├── requirements.txt                               # Required dependencies
└── README.md                                      # This file
```

---

## 🚀 Installation

### 1. Install Dependencies
Requires Python 3.8+ and GPU (recommended):

**Install FFmpeg (Required):**
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

**Install Python Dependencies:**
```bash

git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning-basic

pip install -r requirements.txt
```

### 2. Download Required Models (Required)
This script downloads the necessary base models (`ve`, `s3gen`, `t3`) and default tokenizer. **Must be run before training.**
```bash
python setup.py
```

### 3. Configure Environment
Create a `.env` file or edit `src/config.py` to specify your dataset location and training parameters.

---

## 🏋️ Training (Fine-Tuning)

During training, the script loads the original model weights, **intelligently resizes them** for the new vocabulary size, and initializes new tokens using mean initialization from existing tokens for faster adaptation.

### 1. Dataset Preparation

#### Option A: Using TTS Dataset Generator (Recommended)
We recommend using the [TTS Dataset Generator](https://github.com/gokhaneraslan/tts-dataset-generator) tool to automatically create high-quality datasets from audio or video files.

**Quick Start:**
```bash
# Install the dataset generator
git clone https://github.com/gokhaneraslan/tts-dataset-generator.git
cd tts-dataset-generator
pip install -r requirements.txt

# Generate dataset from your audio/video file
python main.py --file your_audio.mp4 --model large --language en --ljspeech True
```

This will automatically:
- Segment audio into optimal chunks (3-10 seconds)
- Transcribe using Whisper AI
- Generate properly formatted `metadata.csv` and audio files
- Output directly to `MyTTSDataset/` folder in LJSpeech format

**Benefits:**
- Saves hours of manual segmentation and transcription
- Optimizes chunk duration for TTS training
- Handles multiple languages (en, tr, fr, de, es, etc.)
- Works with both audio and video files

#### Option B: Manual Dataset Creation
Your dataset should follow the LJSpeech format with a CSV file:
`filename|raw_text|normalized_text`

Example `metadata.csv`:
```text
recording_001|Hello world.|hello world
recording_002|This is a test recording.|this is a test recording
```

Place your dataset in the `MyTTSDataset/` folder:
```text
MyTTSDataset/
├── metadata.csv
└── wavs/
    ├── recording_001.wav
    ├── recording_002.wav
    └── ...
```

**Dataset Quality Requirements:**
- Sample rate: 16kHz, 22.05kHz, or 44.1kHz (will be resampled to 16kHz automatically)
- Format: WAV (mono or stereo - will be converted to mono automatically)
- Duration: 3-10 seconds per segment (optimal for TTS)
- Minimum total duration: 30+ minutes for basic training
- **Recommended:** 1 hour of clean audio for optimal results
- Audio quality: Clean, minimal background noise

### 2. Configuration
**Important:** Ensure the `NEW_VOCAB_SIZE` in **both** `src/config.py` **AND** `inference.py` matches the number of tokens in your custom `tokenizer.json`.

**For non-English languages:**
1. Create your custom `tokenizer.json` with all characters in your target language
2. Count the total tokens in your JSON file
3. Update `NEW_VOCAB_SIZE` in both files to match this count

Example for Turkish:
```python
# In src/config.py
NEW_VOCAB_SIZE = 2454  # Must match your tokenizer.json

# In inference.py
NEW_VOCAB_SIZE = 2454  # Must be identical to config.py
```

Other key parameters to adjust:
```python
# Dataset
DATASET_PATH = "MyTTSDataset"
METADATA_FILE = "metadata.csv"

# Training
BATCH_SIZE = 4         # Adjust based on your GPU VRAM
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
```

If your dataset is file-based dataset, set ljspeech = False in the configuration file.
```python
# In src/config.py
ljspeech = False  

```

### 3. Start Training
```bash
python train.py
```

The trained model will be saved as `chatterbox_output/t3_finetuned.safetensors`.

**Training Tips:**
*   **VRAM:** T3 is a Transformer model with high VRAM usage. For 12GB VRAM, use `batch_size=4`. For lower VRAM, use `batch_size=2` with `grad_accum=32`.
*   **Mixed Precision:** Code uses `fp16=True` by default for faster training and memory efficiency.
*   **Checkpointing:** Models are saved every epoch in `chatterbox_output/`.
*   **Recommended Training Duration:** For optimal results with 1 hour of target speaker audio, train for **150 epochs** or **1000 steps**. This configuration typically produces high-quality voice cloning results.

---

## 🗣️ Inference (Speech Synthesis)

The inference script loads your fine-tuned `.safetensors` file and uses **Silero VAD** to automatically trim unwanted silence/noise at the end of generated audio.

### 1. Prepare Reference Audio (Prompt)
Chatterbox is a voice cloning/style transfer model. You **must provide a reference `.wav` file** (audio prompt) for inference.

Place your reference audio in `speaker_reference/`:
```text
speaker_reference/
└── reference.wav
```

**Reference Audio Requirements:**
*   Format: WAV, mono or stereo
*   Sample rate: Any (will be resampled automatically)
*   Duration: 3-10 seconds recommended
*   Quality: Clean audio with minimal background noise

### 2. Running Inference
Edit `inference.py` to set your text and audio prompt paths:

```python
TEXT_TO_SAY = "This is a test of the fine-tuned model."
AUDIO_PROMPT = "speaker_reference/reference.wav"
```

Run inference:
```bash
python inference.py
```

The output will be saved as `output_stitched.wav` (24kHz).

### 3. Advanced Usage

**Multiple Sentences:**
The script automatically splits long text into sentences for better quality:
```python
TEXT_TO_SAY = "Hello! How are you today? This is amazing."
```

**Audio Processing:**
All audio is automatically processed to mono and resampled to the correct sample rate using FFmpeg. The output format is:
- **Channels:** Mono (1 channel)
- **Sample Rate:** 24kHz
- **Codec:** 16-bit PCM WAV

---

## 🛠️ Technical Details

### Why Preprocessing?
Original Chatterbox training pipelines often process audio "on-the-fly" (resampling, feature extraction) during training. This causes the GPU to wait for the CPU, slowing down training significantly.
By running `preprocess.py`, we:
1.  Extract Speaker Embeddings (Voice Encoder)
2.  Extract Acoustic Tokens (S3Gen)
3.  Tokenize Text
4.  Save everything as optimized PyTorch tensors (`.pt`)
This allows the `dataset.py` to simply load tensors, maximizing GPU utilization.

### Tokenizer Structure
The `pretrained_models/tokenizer.json` file downloaded by `setup.py` includes support for **23 languages** with extensive grapheme coverage. This file is used by `src/chatterbox/tokenizer.py` during both training and inference.

**Default Multi-Language Support:**
The provided tokenizer already covers common characters from 23 languages, including but not limited to:
- Latin-based languages (English, French, Spanish, German, Italian, Portuguese)
- Turkish with special characters (ç, ğ, ı, ö, ş, ü)
- Eastern European languages
- And more

**When to Create a Custom Tokenizer:**
You only need to create a custom tokenizer if:
1. Your target language has special characters not in the default set
2. You want to optimize the vocab size for a specific language
3. You need to add domain-specific symbols or characters

**Creating a Custom Tokenizer (Optional):**

1. **Identify all characters** in your target language:
   - All letters (including accented/special characters)
   - Numbers (0-9)
   - Punctuation marks
   - Special symbols used in your language

2. **Create the JSON mapping** - Example structure:
```json
{
  "a": 0,
  "b": 1,
  "c": 2,
  "ç": 3,
  "d": 4,
  ...
  " ": 100,
  ".": 101,
  ",": 102,
  ...
}
```

3. **Count total tokens** in your JSON file

4. **Update NEW_VOCAB_SIZE** in both `src/config.py` AND `inference.py` to match the token count

5. **Replace** `pretrained_models/tokenizer.json` with your custom file before training

**Vocab Size Examples:**
- **Default (23 languages):** Check your downloaded `tokenizer.json` for exact count
- **Custom French:** ~200 tokens (if you want French-only optimization)
- **Custom German:** ~180 tokens (if you want German-only optimization)

**Important:** The default tokenizer should work for most languages. Only customize if you have specific requirements or encounter missing characters.

### VAD Integration
During inference, `inference.py` uses Silero VAD to prevent hallucinations and sentence-ending elongations. This automatically trims unwanted silence and noise from generated audio.

### Audio Processing Pipeline
All audio processing uses **FFmpeg** for professional-quality results:
- **Input:** Automatic conversion to mono (1 channel)
- **Resampling:** Automatic resampling to required sample rates
- **Training:** 16kHz processing
- **Output:** 24kHz, 16-bit PCM WAV format
- **Codec:** `pcm_s16le` (16-bit signed little-endian PCM)

### Model Architecture
*   **VE (Voice Encoder):** Extracts speaker embeddings from reference audio
*   **T3 (Text-to-Speech):** Main transformer-based TTS model (this is what you fine-tune)
*   **S3Gen (Vocoder):** Converts mel-spectrograms to waveforms

---

## 📝 Troubleshooting

**Error:** `RuntimeError: Error(s) in loading state_dict for T3... size mismatch`
*   **Solution:** `NEW_VOCAB_SIZE` doesn't match the token count in `tokenizer.json`. 
*   **Check:** 
    1. Count tokens in your `tokenizer.json` file
    2. Verify `NEW_VOCAB_SIZE` in `src/config.py` matches this count
    3. Verify `NEW_VOCAB_SIZE` in `inference.py` also matches (must be identical)
*   **Common mistake:** Updating only one file but not the other

**Error:** `FileNotFoundError: ... ve.safetensors`
*   **Solution:** You haven't downloaded base models. Run `python setup.py`.

**Error:** `CUDA out of memory`
*   **Solution:** Reduce `BATCH_SIZE` in `src/config.py` or enable gradient accumulation.

**Poor Quality Output:**
*   Check reference audio quality (should be clean, at least 5 seconds)
*   Ensure adequate training data (minimum 30 minutes recommended)

---

## 🙏 Acknowledgments

Based on the Chatterbox TTS model architecture. Special thanks to the original authors and contributors.

---

## 📧 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review `src/config.py` for configuration options
3. Open an issue on GitHub with detailed error messages and your setup information
