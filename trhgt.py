import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index

class TRHGTConv(MessagePassing):
    r"""
    Temporal-Relative Heterogeneous Graph Transformer (TR-HGT).
    
    Novelty: 
    Injects a learnable Relative Temporal Bias (Scalar) into the attention mechanism.
    
    Update Analysis (Run 4 Fixes):
    Previous runs showed severe overfitting (Train 88% vs Test 70%). The Vector-based
    temporal encoding ($Q \cdot R$) was too high-capacity for IEMOCAP.
    
    This version switches to **T5-style Scalar Relative Bias**:
    Instead of interacting with content ($Q$), we simply add a learned scalar 
    bias based on distance. This acts as a robust "soft decay" prior that is 
    much harder to overfit.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        max_relative_positions: int = 8, 
        dropout: float = 0.2, 
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"out_channels must be divisible by heads")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {edge_type: i for i, edge_type in enumerate(metadata[1])}
        self.dst_node_types = {key[-1] for key in self.edge_types}
        self.max_relative_positions = max_relative_positions
        self.dropout = dropout

        self.kqv_lin = HeteroDictLinear(self.in_channels, self.out_channels * 3)
        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels, types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)

        # FIX: Switch from Vector Embedding [Max, Heads, Dim] to Scalar Bias [Max, Heads]
        # This drastically reduces parameters to prevent overfitting.
        self.rel_temporal_bias = nn.Parameter(torch.Tensor(max_relative_positions, heads))
        
        # Learnable Gate (Scalar)
        self.time_gate = nn.Parameter(torch.zeros(1))

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1)) for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)
        
        # FIX: Initialize Scalar Bias with Zeros
        # Since this is just a bias term (addition), 0.0 means "Standard HGT Behavior".
        # We don't need noise here because gradients flow directly to the bias.
        nn.init.constant_(self.rel_temporal_bias, 0.0)
        
        # Initialize gate to small positive to allow flow
        nn.init.constant_(self.time_gate, 0.1)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        ks, vs, type_list = [], [], []
        offset: Dict[EdgeType] = {}
        
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        batch_dict: Dict[NodeType, Tensor] = None 
    ) -> Dict[NodeType, Optional[Tensor]]:

        out_dim = self.out_channels
        H = self.heads
        D = out_dim // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(k_dict, v_dict, edge_index_dict)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))

        # --- Temporal Logic ---
        src_idx, dst_idx = edge_index
        dist = (dst_idx - src_idx).abs()
        dist = torch.clamp(dist, max=self.max_relative_positions - 1)
        
        # Retrieve Scalar Bias instead of Vector
        # Shape: [num_edges, heads]
        temporal_bias = self.rel_temporal_bias[dist] 
        
        # Apply Dropout to the bias values
        temporal_bias = F.dropout(temporal_bias, p=self.dropout, training=self.training)

        # 4. Propagate with bias
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr, temporal_bias=temporal_bias)

        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        a_dict = self.out_lin({
            k: torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        for node_type, out in out_dict.items():
            out = a_dict[node_type]
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int], 
                temporal_bias: Tensor) -> Tensor:
        
        # A. Content-based Attention: [E, H]
        content_score = (q_i * k_j).sum(dim=-1)
        
        # B. Temporal Bias (Scalar): [E, H]
        # We no longer do dot products. We just add the bias.
        # This is the "T5 Relative Bias" approach.
        
        # C. Gated Combination
        gate = torch.tanh(self.time_gate)
        
        # Equation: (Content + Gate * Bias) * EdgePrior
        # This modulates the raw attention scores based on distance
        alpha = (content_score + (gate * temporal_bias)) * edge_attr
        
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)