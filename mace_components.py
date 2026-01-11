"""
MACE Neural Network Components

This module contains the pure PyTorch implementation of MACE
(Multi-Atomic Cluster Expansion) for molecular property prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import math

class RadialBasis(nn.Module):
    """Bessel radial basis for distance encoding"""

    def __init__(self, num_basis=8, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.frequencies = nn.Parameter(
            torch.arange(1, num_basis + 1) * math.pi / cutoff,
            requires_grad=False
        )

    def forward(self, distances):
        """Encode distances as basis functions"""
        # Cutoff envelope
        envelope = torch.where(
            distances < self.cutoff,
            torch.cos(distances * math.pi / (2 * self.cutoff)) ** 2,
            torch.zeros_like(distances)
        )

        # Bessel basis
        d = distances.unsqueeze(-1)
        basis = torch.sin(self.frequencies * d) / d

        return basis * envelope.unsqueeze(-1)

print("✅ RadialBasis defined")

class MACEInteraction(MessagePassing):
    """MACE message passing interaction layer"""

    def __init__(self, hidden_dim, num_basis):
        super().__init__(aggr='add')

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        """Message passing step"""
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_net(torch.cat([x, out], dim=-1))
        return out

    def message(self, x_i, x_j, edge_attr):
        """Construct messages"""
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(msg_input)

print("✅ MACEInteraction defined")

class MACE(nn.Module):
    """MACE: Multi-Atomic Cluster Expansion Neural Network"""

    def __init__(
        self,
        num_elements=100,
        hidden_dim=64,
        num_interactions=2,
        num_basis=8,
        cutoff=5.0
    ):
        super().__init__()

        self.cutoff = cutoff

        # Atom embedding
        self.atom_embedding = nn.Embedding(num_elements, hidden_dim)

        # Radial basis
        self.radial_basis = RadialBasis(num_basis, cutoff)

        # Interaction layers
        self.interactions = nn.ModuleList([
            MACEInteraction(hidden_dim, num_basis)
            for _ in range(num_interactions)
        ])

        # Energy head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, pos, edge_index, batch=None):
        """
        Forward pass

        Args:
            z: (N,) atomic numbers
            pos: (N, 3) atomic positions
            edge_index: (2, E) edge indices
            batch: (N,) batch assignment

        Returns:
            energy: (batch_size,) or scalar
            forces: (N, 3) atomic forces
        """
        # Edge features
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_dist = edge_vec.norm(dim=-1)
        edge_attr = self.radial_basis(edge_dist)

        # Embed atoms
        x = self.atom_embedding(z)

        # Message passing with residual connections
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_attr)

        # Predict atomic energies
        atomic_energies = self.energy_head(x)

        # Sum to molecular energy
        if batch is None:
            energy = atomic_energies.sum()
        else:
            from torch_geometric.utils import scatter
            energy = scatter(atomic_energies, batch, dim=0, reduce='sum')

        energy = energy.squeeze(-1)

        # Compute forces
        forces = None
        if pos.requires_grad:
            forces = -torch.autograd.grad(
                energy.sum(), pos, create_graph=True
            )[0]

        return energy, forces

print("✅ MACE neural network defined")