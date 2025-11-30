# -*- coding: utf-8 -*-
"""
MELD-specific hyper-params & paths
"""
import torch
from pathlib import Path

# ─── Data ────────────────────────────────────────────────────────────────
# DATA_PATH   = Path("/home/user/MER/MELD/pkl_file/meld_multi_features.pkl")

DATA_PATH   = Path("meld_multi_features.pkl")
NUM_CLASSES = 7
EMOTIONS    = ["neutral","surprise","fear","sadness","joy","disgust","anger"]

FEATURE_DIMS = dict(text=1024, audio=300, visual=342)
TEXT_FEATURE_KEY = "videoText1"

# ─── Model ───────────────────────────────────────────────────────────────
HGT_HIDDEN_CHANNELS = 256
HGT_NUM_HEADS       = 4
HGT_NUM_LAYERS      = 3
DROPOUT_RATE        = 0.1849

# ─── Optim / training ────────────────────────────────────────────────────
LEARNING_RATE = 1e-5
BATCH_SIZE     = 16
EPOCHS         = 50
WEIGHT_DECAY   = 0.0013088

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT="ICLR_MELD"

NODE_TYPES = ["audio", "text", "visual"]
EDGE_TYPES = [
    # Speaker-aware temporal edges (unidirectional, past context only)
    ("text", "past_same", "text"),
    ("text", "past_other", "text"),
    ("audio", "past_same", "audio"),
    ("audio", "past_other", "audio"),
    ("visual", "past_same", "visual"),
    ("visual", "past_other", "visual"),
    
    # Cross-modal edges (remain the same)
    ("text","text_to_audio","audio"),  
    ("audio","audio_to_text","text"),
    ("text","text_to_visual","visual"),
    ("visual","visual_to_text","text"),
    ("audio","audio_to_visual","visual"),
    ("visual","visual_to_audio","audio"),
]


# EDGE_TYPES = [
#     # --- Temporal Edges (Windowed 3 past, 3 future) ---
#     # Past context
#     ("text", "past_same", "text"),
#     ("text", "past_other", "text"),
#     ("audio", "past_same", "audio"),
#     ("audio", "past_other", "audio"),
#     ("visual", "past_same", "visual"),
#     ("visual", "past_other", "visual"),
    
#     # Future context (NEW)
#     ("text", "future_same", "text"),
#     ("text", "future_other", "text"),
#     ("audio", "future_same", "audio"),
#     ("audio", "future_other", "audio"),
#     ("visual", "future_same", "visual"),
#     ("visual", "future_other", "visual"),
    
#     # --- Cross-modal edges ---
#     ("text","text_to_audio","audio"),  
#     ("audio","audio_to_text","text"),
#     ("text","text_to_visual","visual"),
#     ("visual","visual_to_text","text"),
#     ("audio","audio_to_visual","visual"),
#     ("visual","visual_to_audio","audio"),
# ]


# EDGE_TYPES = [
#     # --- Temporal Edges (Bidirectional, Windowed) ---
#     # "temporal_same"  = connection to same speaker (past or future)
#     # "temporal_other" = connection to different speaker (past or future)
#     ("text", "temporal_same", "text"),
#     ("text", "temporal_other", "text"),
#     ("audio", "temporal_same", "audio"),
#     ("audio", "temporal_other", "audio"),
#     ("visual", "temporal_same", "visual"),
#     ("visual", "temporal_other", "visual"),
    
#     # --- Cross-modal edges (Unchanged) ---
#     ("text","text_to_audio","audio"),  
#     ("audio","audio_to_text","text"),
#     ("text","text_to_visual","visual"),
#     ("visual","visual_to_text","text"),
#     ("audio","audio_to_visual","visual"),
#     ("visual","visual_to_audio","audio"),
# ]