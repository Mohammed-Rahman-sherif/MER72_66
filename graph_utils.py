# # graph_utils.py (the recommended new version)

# from __future__ import annotations
# import torch
# from torch_geometric.data import HeteroData

# def build_hetero_graph_for_dialogue(
#         text_f, audio_f, visual_f, labels, speakers, vid, # <-- Added 'speakers'
#         *, node_types):
#     """
#     Builds a single graph for an entire dialogue.
#     Nodes represent utterances.
#     This version uses speaker-aware temporal edges.
#     """
#     data = HeteroData()
#     feats = dict(text=text_f, audio=audio_f, visual=visual_f)

#     for nt, feat in feats.items():
#         data[nt].x = feat

#     num_utterances = next(iter(data.x_dict.values())).shape[0]

#     # --- NEW: Speaker-aware temporal edges (DialogueGCN style) ---
#     # This creates unidirectional edges based on speaker turn dynamics.
    
#     # Edges for utterances from the SAME speaker
#     same_speaker_src, same_speaker_dst = [], []
#     # Edges for utterances from a DIFFERENT speaker
#     other_speaker_src, other_speaker_dst = [], []

#     # For each utterance 'j' (destination)...
#     for j in range(num_utterances):
#         # ...look at all previous utterances 'i' (source).
#         for i in range(j):
#             # If the speaker is the same, add a 'past_same' edge
#             if speakers[i] == speakers[j]:
#                 same_speaker_src.append(i)
#                 same_speaker_dst.append(j)
#             # If the speaker is different, add a 'past_other' edge
#             else:
#                 other_speaker_src.append(i)
#                 other_speaker_dst.append(j)
    
#     # Add these new edge types to the graph for each modality
#     for nt in data.node_types:
#         # Add 'past_same' edges
#         edge_index_same = torch.tensor([same_speaker_src, same_speaker_dst], dtype=torch.long)
#         data[nt, 'past_same', nt].edge_index = edge_index_same
        
#         # Add 'past_other' edges
#         edge_index_other = torch.tensor([other_speaker_src, other_speaker_dst], dtype=torch.long)
#         data[nt, 'past_other', nt].edge_index = edge_index_other

#     # --- Cross-modal edges (unconditional) ---
#     # This section remains unchanged.
#     nodes = torch.arange(num_utterances)
#     edge_index = torch.stack([nodes, nodes], 0)
#     pairs = [("text", "audio"), ("text", "visual"), ("audio", "visual")]
#     for mod_a, mod_b in pairs:
#         data[mod_a, f"{mod_a}_to_{mod_b}", mod_b].edge_index = edge_index
#         data[mod_b, f"{mod_b}_to_{mod_a}", mod_a].edge_index = edge_index

#     data.y = labels
#     data.vid = vid
#     return data



# SOTA
from __future__ import annotations
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph_for_dialogue(
        text_f, audio_f, visual_f, labels, speakers, vid,
        *, node_types):
    """
    Builds a single graph for an entire dialogue.
    Nodes represent utterances.
    This version uses speaker-aware temporal edges.
    """
    data = HeteroData()
    feats = dict(text=text_f, audio=audio_f, visual=visual_f)

    for nt, feat in feats.items():
        data[nt].x = feat

    num_utterances = next(iter(data.x_dict.values())).shape[0]

    # --- Speaker-aware temporal edges (DialogueGCN style) ---
    # This creates unidirectional edges based on speaker turn dynamics.
    
    # Edges for utterances from the SAME speaker
    same_speaker_src, same_speaker_dst = [], []
    # Edges for utterances from a DIFFERENT speaker
    other_speaker_src, other_speaker_dst = [], []

    # For each utterance 'j' (destination)...
    for j in range(num_utterances):
        # ...look at all previous utterances 'i' (source).
        for i in range(j):
            # If the speaker is the same, add a 'past_same' edge
            if speakers[i] == speakers[j]:
                same_speaker_src.append(i)
                same_speaker_dst.append(j)
            # If the speaker is different, add a 'past_other' edge
            else:
                other_speaker_src.append(i)
                other_speaker_dst.append(j)
    
    # Add these new edge types to the graph for each modality
    for nt in data.node_types:
        # Add 'past_same' edges if any exist
        if same_speaker_src:
            edge_index_same = torch.tensor([same_speaker_src, same_speaker_dst], dtype=torch.long)
            data[nt, 'past_same', nt].edge_index = edge_index_same
        else:
            data[nt, 'past_same', nt].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Add 'past_other' edges if any exist
        if other_speaker_src:
            edge_index_other = torch.tensor([other_speaker_src, other_speaker_dst], dtype=torch.long)
            data[nt, 'past_other', nt].edge_index = edge_index_other
        else:
            data[nt, 'past_other', nt].edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- Cross-modal edges (unconditional) ---
    nodes = torch.arange(num_utterances)
    edge_index = torch.stack([nodes, nodes], 0)
    pairs = [("text", "audio"), ("text", "visual"), ("audio", "visual")]
    for mod_a, mod_b in pairs:
        data[mod_a, f"{mod_a}_to_{mod_b}", mod_b].edge_index = edge_index
        data[mod_b, f"{mod_b}_to_{mod_a}", mod_a].edge_index = edge_index

    data.y = labels
    data.vid = vid
    return data




# from __future__ import annotations
# import torch
# from torch_geometric.data import HeteroData

# def build_hetero_graph_for_dialogue(
#         text_f, audio_f, visual_f, labels, speakers, vid,
#         *, node_types, window_past=3, window_future=3): # Added window args
#     """
#     Builds a single graph for an entire dialogue with bidirectional windowed context.
#     Matches GraphSmile's topology: connects past and future utterances within a window.
#     """
#     data = HeteroData()
#     feats = dict(text=text_f, audio=audio_f, visual=visual_f)

#     for nt, feat in feats.items():
#         data[nt].x = feat

#     num_utterances = next(iter(data.x_dict.values())).shape[0]

#     # Containers for edge indices
#     # (src, dst)
#     edges = {
#         'past_same':   ([], []),
#         'past_other':  ([], []),
#         'future_same': ([], []),
#         'future_other':([], [])
#     }

#     # Iterate over every utterance 'j' (the destination/target node)
#     for j in range(num_utterances):
        
#         # --- 1. PAST Context (window_past) ---
#         # Look at utterances 'i' BEFORE 'j'
#         start_past = max(0, j - window_past)
#         for i in range(start_past, j):
#             if speakers[i] == speakers[j]:
#                 edges['past_same'][0].append(i)
#                 edges['past_same'][1].append(j)
#             else:
#                 edges['past_other'][0].append(i)
#                 edges['past_other'][1].append(j)

#         # --- 2. FUTURE Context (window_future) ---
#         # Look at utterances 'k' AFTER 'j' (Information flows from Future -> Current)
#         end_future = min(num_utterances, j + window_future + 1)
#         for k in range(j + 1, end_future):
#             if speakers[k] == speakers[j]:
#                 edges['future_same'][0].append(k)
#                 edges['future_same'][1].append(j)
#             else:
#                 edges['future_other'][0].append(k)
#                 edges['future_other'][1].append(j)
    
#     # Add these edge types to the graph for each modality
#     for nt in data.node_types:
#         for edge_type, (src_list, dst_list) in edges.items():
#             if src_list:
#                 edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
#                 data[nt, edge_type, nt].edge_index = edge_index
#             else:
#                 # Create empty edge index if no edges of this type exist
#                 data[nt, edge_type, nt].edge_index = torch.empty((2, 0), dtype=torch.long)

#     # --- Cross-modal edges (unconditional) ---
#     # These remain fully connected (or 1-to-1) as before
#     nodes = torch.arange(num_utterances)
#     edge_index = torch.stack([nodes, nodes], 0)
#     pairs = [("text", "audio"), ("text", "visual"), ("audio", "visual")]
#     for mod_a, mod_b in pairs:
#         data[mod_a, f"{mod_a}_to_{mod_b}", mod_b].edge_index = edge_index
#         data[mod_b, f"{mod_b}_to_{mod_a}", mod_a].edge_index = edge_index

#     data.y = labels
#     data.vid = vid
#     return data




# from __future__ import annotations
# import torch
# from torch_geometric.data import HeteroData

# def build_hetero_graph_for_dialogue(
#         text_f, audio_f, visual_f, labels, speakers, vid,
#         *, node_types, window_past=3, window_future=3):
#     """
#     Builds a single graph for an entire dialogue.
#     Merges past and future connections into 'temporal_same' and 'temporal_other'.
#     """
#     data = HeteroData()
#     feats = dict(text=text_f, audio=audio_f, visual=visual_f)

#     for nt, feat in feats.items():
#         data[nt].x = feat

#     num_utterances = next(iter(data.x_dict.values())).shape[0]

#     # Containers for edge indices
#     # We now only have two types of temporal edges
#     edges = {
#         'temporal_same':  ([], []), # Same speaker (past or future)
#         'temporal_other': ([], [])  # Different speaker (past or future)
#     }

#     # Iterate over every utterance 'j' (the destination/target node)
#     for j in range(num_utterances):
        
#         # --- 1. PAST Context ---
#         start_past = max(0, j - window_past)
#         for i in range(start_past, j):
#             if speakers[i] == speakers[j]:
#                 edges['temporal_same'][0].append(i)
#                 edges['temporal_same'][1].append(j)
#             else:
#                 edges['temporal_other'][0].append(i)
#                 edges['temporal_other'][1].append(j)

#         # --- 2. FUTURE Context ---
#         end_future = min(num_utterances, j + window_future + 1)
#         for k in range(j + 1, end_future):
#             if speakers[k] == speakers[j]:
#                 edges['temporal_same'][0].append(k)
#                 edges['temporal_same'][1].append(j)
#             else:
#                 edges['temporal_other'][0].append(k)
#                 edges['temporal_other'][1].append(j)
    
#     # Add these edge types to the graph for each modality
#     for nt in data.node_types:
#         for edge_type, (src_list, dst_list) in edges.items():
#             if src_list:
#                 edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
#                 data[nt, edge_type, nt].edge_index = edge_index
#             else:
#                 data[nt, edge_type, nt].edge_index = torch.empty((2, 0), dtype=torch.long)

#     # --- Cross-modal edges (Same as before) ---
#     nodes = torch.arange(num_utterances)
#     edge_index = torch.stack([nodes, nodes], 0)
#     pairs = [("text", "audio"), ("text", "visual"), ("audio", "visual")]
#     for mod_a, mod_b in pairs:
#         data[mod_a, f"{mod_a}_to_{mod_b}", mod_b].edge_index = edge_index
#         data[mod_b, f"{mod_b}_to_{mod_a}", mod_a].edge_index = edge_index

#     data.y = labels
#     data.vid = vid
#     return data