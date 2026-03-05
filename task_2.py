import os
import random
from collections import Counter
from typing import Any, Dict, List

import numpy as np
import torch
import evaluate
import librosa
from scipy.signal import fftconvolve

from datasets import load_dataset, Audio
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# ================================
# Cache setup
# ================================
CACHE_ROOT = "/kaggle/working/cache"
os.makedirs(CACHE_ROOT, exist_ok=True)
os.environ["HF_HOME"]            = CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_ROOT, "transformers")
os.environ["HF_DATASETS_CACHE"]  = os.path.join(CACHE_ROOT, "datasets")
os.environ["TORCH_HOME"]         = os.path.join(CACHE_ROOT, "torch")

# ================================
# Reproducibility
# ================================
set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name())

# ================================
# Model
# ================================
model_id = "facebook/wav2vec2-xls-r-300m"

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=True,
)

# ================================
# Load dataset
# ================================
dataset  = load_dataset("badrex/nnti-dataset-full")
train_ds = dataset["train"].shuffle(seed=42)
valid_ds = dataset["validation"].shuffle(seed=42)

train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

LABELS     = train_ds.unique("language")
str_to_int = {s: i for i, s in enumerate(LABELS)}
int_to_str = {i: s for s, i in str_to_int.items()}
num_labels = len(LABELS)

print("Number of languages:", num_labels)
print("Class distribution:", Counter(train_ds["language"]))

# ================================
# TASK 2: AUGMENTATION FUNCTIONS
# Pure librosa + numpy — no audiomentations version issues
#
# Why each augmentation breaks speaker-language shortcut:
# 1. pitch_shift   → changes F0 (fundamental freq) — core of speaker identity
# 2. time_stretch  → changes speaking rate — alters temporal speaker patterns
# 3. add_noise     → forces phoneme focus, not clean speaker fingerprint
# 4. time_shift    → breaks utterance start-position memorisation
# 5. apply_reverb  → room acoustics — speaker sounds different in each room
# ================================

def pitch_shift(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Shift pitch by random semitones — breaks speaker F0 identity."""
    n_steps = random.uniform(-3.0, 3.0)
    return librosa.effects.pitch_shift(
        signal, sr=sr, n_steps=n_steps
    ).astype(np.float32)


def time_stretch(signal: np.ndarray) -> np.ndarray:
    """Stretch/compress time — breaks temporal speaker patterns."""
    rate = random.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(signal, rate=rate)
    # Keep original length: pad or trim
    if len(stretched) > len(signal):
        stretched = stretched[:len(signal)]
    else:
        stretched = np.pad(stretched, (0, len(signal) - len(stretched)))
    return stretched.astype(np.float32)


def add_noise(signal: np.ndarray) -> np.ndarray:
    """Add Gaussian noise — forces phoneme focus over speaker fingerprint."""
    amplitude = random.uniform(0.001, 0.015)
    noise     = np.random.randn(len(signal)).astype(np.float32) * amplitude
    return (signal + noise).astype(np.float32)


def time_shift(signal: np.ndarray) -> np.ndarray:
    """Random time shift — breaks start-of-utterance memorisation."""
    shift = int(random.uniform(-0.1, 0.1) * len(signal))
    return np.roll(signal, shift).astype(np.float32)


def apply_reverb(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Synthetic room impulse response — speaker sounds different per room."""
    try:
        rt60      = random.uniform(0.1, 0.8)
        ir_length = int(rt60 * sr)
        if ir_length < 1:
            return signal
        t  = np.linspace(0, rt60, ir_length)
        ir = np.random.randn(ir_length) * np.exp(-6.9 * t / rt60)
        ir[0] = 1.0
        out = fftconvolve(signal, ir)[:len(signal)]
        if np.max(np.abs(out)) > 0:
            out = out / np.max(np.abs(out))
        return out.astype(np.float32)
    except Exception:
        return signal


def augment(signal: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Apply all augmentations with individual probabilities."""
    # Normalise first
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal))

    if random.random() < 0.7:   # PitchShift — most important for speaker bias
        signal = pitch_shift(signal, sr)
    if random.random() < 0.5:   # TimeStretch
        signal = time_stretch(signal)
    if random.random() < 0.4:   # Noise
        signal = add_noise(signal)
    if random.random() < 0.4:   # TimeShift
        signal = time_shift(signal)
    if random.random() < 0.3:   # Reverb
        signal = apply_reverb(signal, sr)

    return signal


max_duration = 7

# ================================
# Preprocessing
# ================================
def preprocess_train(examples):
    processed = []
    for audio in examples["audio_filepath"]:
        signal = np.array(audio["array"], dtype=np.float32)
        signal = augment(signal, sr=16000)          # apply all augmentations
        processed.append(signal)

    inputs = feature_extractor(
        processed,
        sampling_rate=16000,
        truncation=True,
        max_length=16000 * max_duration,
        return_attention_mask=True,
    )
    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    return inputs


def preprocess_valid(examples):
    # NO augmentation on validation — measure true performance
    processed = [
        np.array(audio["array"], dtype=np.float32)
        for audio in examples["audio_filepath"]
    ]
    inputs = feature_extractor(
        processed,
        sampling_rate=16000,
        truncation=True,
        max_length=16000 * max_duration,
        return_attention_mask=True,
    )
    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    return inputs


train_encoded = train_ds.map(
    preprocess_train,
    remove_columns=train_ds.column_names,
    batched=True,
    batch_size=32,
)
valid_encoded = valid_ds.map(
    preprocess_valid,
    remove_columns=valid_ds.column_names,
    batched=True,
    batch_size=32,
)

# ================================
# Model config
# ================================
config            = AutoConfig.from_pretrained(model_id)
config.num_labels = num_labels
config.label2id   = str_to_int
config.id2label   = int_to_str

model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    config=config,
    ignore_mismatched_sizes=True,
)

for param in model.wav2vec2.parameters():
    param.requires_grad = False
print("Phase 1: Encoder frozen — training classification head only")

# ================================
# Data collator
# ================================
class AudioDataCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]):
        batch = self.feature_extractor.pad(
            {
                "input_values":   [f["input_values"]   for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(
            [f["label"] for f in features], dtype=torch.long
        )
        return batch

data_collator = AudioDataCollator(feature_extractor)

# ================================
# Metrics
# ================================
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(
        predictions=preds,
        references=eval_pred.label_ids,
    )

# ================================
# PHASE 1: Frozen encoder — train head only
# ================================
training_args = TrainingArguments(
    output_dir="/kaggle/working/slid_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,            # reduced — head converges fast
    learning_rate=1e-4,             # FIXED: was 1e-3 (too high → diverging loss)
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=2,       # reduced — Kaggle sometimes errors at 4
    group_by_length=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=valid_encoded,
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # was 3 — too aggressive
)

print("\nPhase 1: Starting training (frozen encoder)...")
trainer.train()

# ================================
# PHASE 2: Unfreeze encoder, fine-tune at lower LR
# ================================
print("\nPhase 2: Unfreezing encoder for full fine-tuning...")
for param in model.wav2vec2.parameters():
    param.requires_grad = True

training_args_phase2 = TrainingArguments(
    output_dir="/kaggle/working/slid_model_phase2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=15,            # more epochs for full fine-tune
    learning_rate=3e-5,             # FIXED: was 1e-5 (too slow to converge)
    weight_decay=0.01,
    warmup_ratio=0.15,              # longer warmup for full model
    lr_scheduler_type="cosine",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=2,
    group_by_length=True,
)

trainer_phase2 = Trainer(
    model=model,
    args=training_args_phase2,
    train_dataset=train_encoded,
    eval_dataset=valid_encoded,
    tokenizer=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer_phase2.train()

# ================================
# Final evaluation & save
# ================================
print("\nFinal evaluation...")
metrics = trainer_phase2.evaluate()
print(metrics)

save_dir = "/kaggle/working/slid_model_final"
model.save_pretrained(save_dir)
feature_extractor.save_pretrained(save_dir)
print(f"✓ Model saved to {save_dir}")

import shutil
shutil.make_archive("/kaggle/working/language_id_model", "zip", save_dir)
print("✓ Zipped: /kaggle/working/language_id_model.zip")