import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from helpers import RBFExpansion,MLP

class EdgeGatedGraphConvolution(nn.Module):
    def __init__(self, hidden_features: int,dropout_prob = 0.25, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.silu = nn.SiLU()
        self.node_Norm = nn.BatchNorm1d(hidden_features)
        self.z_Norm = nn.BatchNorm1d(hidden_features)
        self.W_gate = nn.Linear(3 * hidden_features, hidden_features)
        self.W_src = nn.Linear(hidden_features, hidden_features)
        self.W_dst = nn.Linear(hidden_features, hidden_features)
        self.epsilon = 0.000001
        self.dropout_w_gate =  nn.Dropout(0)
        self.dropout_w_src =  nn.Dropout(0)
        self.dropout_w_dst =  nn.Dropout(0)

    def edgegate(self, src, dst, node_features, edge_features) -> torch.Tensor:
        src_node_features = node_features[src]
        dst_node_features = node_features[dst]
        z = torch.cat(
            (src_node_features, dst_node_features, edge_features),
            dim=1,
        )
        return self.dropout_w_gate(self.silu(self.z_Norm(self.W_gate(z))))

    def normalize_edges(self, src, edge_features) -> torch.Tensor:
        sigmoid = torch.sigmoid(edge_features)
        aggregated_edge_features = scatter(sigmoid, index=src.to(torch.long))
        denominator = (
            aggregated_edge_features[src] + self.epsilon
        )  # Epsilon in equation 3

        return sigmoid / denominator

    def aggregate_nodes(self, src, dst, normalized_edge_features, node_features):
        updated_src_node_ft = self.W_src(node_features.to(torch.float))
        updated_src_node_ft = self.dropout_w_src(updated_src_node_ft)
        thee = self.W_dst(node_features.to(torch.float))[dst]
        thee = self.dropout_w_dst(thee)
        updated_dst_node_ft = scatter(normalized_edge_features * thee, src)
        return updated_src_node_ft + updated_dst_node_ft

    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        src, dst = edge_index.to(torch.int64)
        updated_edge_features = self.edgegate(src, dst, node_features, edge_features)
        normalized_edge_features = self.normalize_edges(src, updated_edge_features)
        updated_node_features = node_features+self.silu(
            self.node_Norm(
                self.aggregate_nodes(src, dst, normalized_edge_features, node_features)
            )
        )
        return updated_node_features, updated_edge_features
    
class ALIGNNLayer(nn.Module):
    def __init__(self, hidden_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.egcn_atomistic = EdgeGatedGraphConvolution(
            hidden_features
        )
        self.egcn_linegraph = EdgeGatedGraphConvolution(
            hidden_features
        )

    def forward(self, g_edge_index, lg_edge_index, atom_ft, bond_ft, angle_ft):
        m, t = self.egcn_linegraph(
            lg_edge_index,  # Connectivity of line graph
            bond_ft,  # Node features of line graph equivalent to edge features of atomistic graph "
            angle_ft,  # Edge features of line graph "bond angles"
        )
        h, e = self.egcn_atomistic(
            g_edge_index,  # connectivity of atomistic graph
            atom_ft,  # node features of atomistic graph [electronegativity,group number, covalent raduis etc..] in CGCNN style
            m,
        )
        return h, e, t
    

class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config_alignn_layers: int = 2
        self.config_gcn_layers: int = 2
        self.config_atom_input_features: int = 92
        self.config_edge_input_features: int = 80
        self.config_triplet_input_features: int = 40
        self.config_embedding_features: int = 64
        self.config_hidden_features: int = 256
        self.config_link: str = "identity"
        self.config_zero_inflated: bool = False
        self.config_classification: bool = False
        self.config_num_classes: int = 2
        self.config_output_features: int = 1

        self.atom_embedding = MLP(
            self.config_atom_input_features, self.config_hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=self.config_edge_input_features,
            ),
            MLP(self.config_edge_input_features, self.config_embedding_features),
            MLP(self.config_embedding_features, self.config_hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=self.config_triplet_input_features,
            ),
            MLP(self.config_triplet_input_features, self.config_embedding_features),
            MLP(self.config_embedding_features, self.config_hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNLayer(self.config_hidden_features)
                for idx in range(self.config_alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConvolution(self.config_hidden_features)
                for idx in range(self.config_gcn_layers)
            ]
        )

        self.fc = nn.Linear(self.config_hidden_features, self.config_output_features)

    def forward(self, g: Batch, lg: Batch):

        atom_ft = g.x
        angle_ft = lg.edge_attr
        bondlength = torch.norm(g.edge_attr, dim=1)

        atom_ft = self.atom_embedding(atom_ft)
        bond_ft = self.edge_embedding(bondlength)
        angle_ft = self.angle_embedding(angle_ft)

        # ALIGNN Layers
        for alignn_layer in self.alignn_layers:
            atom_ft, bond_ft, angle_ft = alignn_layer(
                g.edge_index, lg.edge_index, atom_ft, bond_ft, angle_ft
            )

        # Normal EGCN Layers
        for egcn_layer in self.gcn_layers:
            atom_ft, bond_ft = egcn_layer(g.edge_index, atom_ft, bond_ft)

        h = global_mean_pool(atom_ft, batch=g.batch)
        out = self.fc(h)
        return out