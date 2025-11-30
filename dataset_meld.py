# In dataset_meld.py

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config_meld as cfg
from graph_utils import build_hetero_graph_for_dialogue

class MELDHeteroDataset(Dataset):
    """
    Dialogue-level MELD loader for a graph-based model.
    This version correctly handles the one-hot encoded speaker format
    found in the MELD pickle file.
    """
    def __init__(self, path, dataset, split='train', split_ratio=0.1, random_seed=33, text_feature_key='videoText1'):
        self.dataset = dataset
        if self.dataset == 'meld':
            self.cfg = cfg
            self.cfg.TEXT_FEATURE_KEY = text_feature_key
            (
                self.videoIDs, self.videoSpeakers, self.videoLabels, _,
                self.videoText1, self.videoText2, self.videoText3, self.videoText4,
                self.videoAudio, self.videoVisual, _,
                self.trainVid, self.testVid, _
            ) = pickle.load(open(path, "rb"), encoding="latin1")
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # --- NEW Speaker Handling ---
        # The speakers are already one-hot encoded. We just need to know the number of speakers.
        # We can determine this from the length of the first speaker vector.
        first_dialogue_id = next(iter(self.videoIDs))
        first_utterance_speaker_vector = self.videoSpeakers[first_dialogue_id][0]
        self.num_speakers = len(first_utterance_speaker_vector)
        print(f"Detected {self.num_speakers} unique speakers from one-hot vector dimension.")
        
        # We no longer need to build a speaker_to_idx map.
        
        # Dialogue key splitting remains the same
        train_keys, val_keys = train_test_split(list(self.trainVid), test_size=split_ratio, random_state=random_seed)

        if split == 'train':
            self.keys = train_keys
        elif split == 'val':
            self.keys = val_keys
        elif split == 'test':
            self.keys = list(self.testVid)
        else:
            raise ValueError(f"Invalid split name: {split}. Choose from 'train', 'val', 'test'.")

        num_dialogues = len(self.keys)
        num_utterances = sum(len(self.videoLabels[vid]) for vid in self.keys)
        print(f"{split.capitalize()}-{self.dataset.upper()}: {num_utterances} utterances from {num_dialogues} dialogues")

    def __len__(self):
        return len(self.keys)

    def _pick_text_matrix(self, vid):
        return getattr(self, self.cfg.TEXT_FEATURE_KEY)[vid]

    def __getitem__(self, idx):
        vid = self.keys[idx]
        
        # Load all feature and label data for the dialogue
        txt = torch.as_tensor(np.array(self._pick_text_matrix(vid)), dtype=torch.float32)
        aud = torch.as_tensor(np.array(self.videoAudio[vid]), dtype=torch.float32)
        vis = torch.as_tensor(np.array(self.videoVisual[vid]), dtype=torch.float32)
        labels = torch.as_tensor(self.videoLabels[vid], dtype=torch.long)
        
        # --- THE FIX: Convert one-hot vectors to integer indices ---
        
        # 1. Load the list of one-hot speaker vectors for the dialogue.
        speakers_one_hot = self.videoSpeakers[vid]
        
        # 2. Convert each one-hot vector to an integer index (0, 1, 2, ...).
        #    np.argmax finds the index of the '1' in each vector.
        speaker_indices = torch.tensor([np.argmax(ohe) for ohe in speakers_one_hot], dtype=torch.long)
        
        # 3. The graph-building function needs a list of identifiers to compare
        #    for `past_same` vs. `past_other` edges. This list of integers works perfectly.
        speaker_ids_for_graph = speaker_indices.tolist()

        # Build the graph for the whole dialogue. All data shapes are now consistent.
        graph = build_hetero_graph_for_dialogue(
                txt, aud, vis, labels, speaker_ids_for_graph, vid,
                node_types=self.cfg.NODE_TYPES
        )
        
        # Attach the final tensor of speaker indices that the model's nn.Embedding layer expects.
        graph.speaker_idx = speaker_indices
        return graph