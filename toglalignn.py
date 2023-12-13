import torch
import myphtest as ph
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from helpers import MLP,Gaussian_transform,RBFExpansion
from alignn import EdgeGatedGraphConvolution,ALIGNNLayer

class TopologyLayer(torch.nn.Module):
    """Topological Aggregation Layer."""

    def __init__(self, features_in, features_out, filtration_hidden, num_filtrations=5):
        """
        num_coord_funs is a dictionary with the numbers of coordinate functions of each type.
        dim1 is a boolean. True if we have to return dim1 persistence.
        """
        super().__init__()
        self.features_in = features_in
        self.features_out = features_out
        self.num_filtrations = num_filtrations
        self.filtration_hidden = filtration_hidden
        self.filtration_modules = torch.nn.Sequential(
            torch.nn.Linear(self.features_in, self.filtration_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(self.filtration_hidden, num_filtrations),
        )
        self.silu = torch.nn.SiLU()
        self.coord_activation0 = Gaussian_transform(2 * num_filtrations)
        # self.coord_activation1 = Gaussian_transform(2 * num_filtrations)

        in_out_dim = self.features_in + self.num_filtrations * 2
        self.out = torch.nn.Linear(in_out_dim, features_out)

    def compute_persistence(
        self,
        x,
        edge_index,
        slice_dict,
    ):
        """
        Returns the persistence pairs as a list of tensors with shape [X.shape[0],2].
        The lenght of the list is the number of filtrations.
        """

        filtered_v_ = self.filtration_modules(x)
        filtered_e_, _ = torch.max(
            torch.stack((filtered_v_[edge_index[0]], filtered_v_[edge_index[1]])),
            axis=0,
        )

        vertex_slices = torch.Tensor(slice_dict["x"]).long()
        edge_slices = torch.Tensor(slice_dict["edge_index"]).long()

        vertex_slices = vertex_slices.cpu()
        edge_slices = edge_slices.cpu()

        filtered_v_ = filtered_v_.cpu().transpose(1, 0).contiguous()
        filtered_e_ = filtered_e_.cpu().transpose(1, 0).contiguous()
        edge_index = edge_index.cpu().transpose(1, 0).contiguous()

        p0, p1 = ph.compute_persistence_homology_batched_mt(
            filtered_v_, filtered_e_, edge_index, vertex_slices, edge_slices
        )
        p0 = torch.cat([x for x in p0], dim=1).to(x.device)
        p1 = torch.cat([x for x in p1], dim=1).to(x.device)

        return p0, p1

    def collapse_dim1(self, activations, mask, slices):
        """
        Takes a flattened tensor of activations along with a mask and collapses it (sum) to have a graph-wise features
        Inputs :
        * activations [N_edges,d]
        * mask [N_edge]
        * slices [N_graphs]
        Output:
        * collapsed activations [N_graphs,d]
        """
        collapsed_activations = []
        for el in range(len(slices) - 1):
            activations_el_ = activations[slices[el] : slices[el + 1]]
            mask_el = mask[slices[el] : slices[el + 1]]
            activations_el = activations_el_[mask_el].sum(axis=0)
            collapsed_activations.append(activations_el)
        return torch.stack(collapsed_activations)

    def forward(self, x,edge_index,slice_dict):
        p0, p1 = self.compute_persistence(x,edge_index,slice_dict)
        p0 = self.coord_activation0(p0)
        mask = (p1 != 0).any(1)
        graph_level_activation = self.collapse_dim1(p1, mask, slice_dict["edge_index"])
        concat_activations = torch.cat((x, p0), 1)      
        out_activations = self.silu(self.out(concat_activations))
        return out_activations,graph_level_activation
    

class TOPOALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config_alignn_layers: int = 2
        self.config_gcn_layers: int = 2
        self.config_edge_input_features: int = 80
        self.config_triplet_input_features: int = 40
        self.config_embedding_features: int = 128
        self.config_hidden_features: int = 256
        self.config_link: str = "identity"
        self.config_zero_inflated: bool = False
        self.config_classification: bool = False
        self.config_num_classes: int = 2
        self.config_output_features: int = 1
        self.config_output_embed_features = 64
        
        self.config_atom_input_features: int = 92
        self.config_num_filtrations = 64
        self.config_topology_hidden = 64
        self.config_topology_output = 256
            
        self.graph_topology = TopologyLayer(self.config_atom_input_features,self.config_topology_output,self.config_topology_hidden,self.config_num_filtrations)
        self.atom_embedding = MLP(
            self.config_topology_output, self.config_hidden_features
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
        self.silu = nn.SiLU()
        self.fc1 = nn.Linear(self.config_hidden_features+self.config_num_filtrations*2, self.config_output_embed_features)
        self.fc2 = nn.Linear(self.config_output_embed_features,self.config_output_features)

    def forward(self, g: Batch, lg: Batch):
        # TODO Make graph and line graph local variable
        atom_ft = g.x
        angle_ft = lg.edge_attr
        bondlength = torch.norm(g.edge_attr, dim=1)
         # Topology Layer
        atom_ft,activations = self.graph_topology(atom_ft,g.edge_index,g._slice_dict)
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
        h = torch.cat([h,activations],dim = 1)
        out = self.fc2(self.silu(self.fc1(h)))
        return out
    
