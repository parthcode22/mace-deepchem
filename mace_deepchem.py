"""
MACE-DeepChem Integration

This module wraps the MACE neural network as a DeepChem TorchModel,
enabling seamless integration with DeepChem's training and evaluation pipeline.
"""

from deepchem.models import TorchModel
from deepchem.models.losses import Loss
from torch_geometric.data import Data, Batch
import torch
import numpy as np

from mace_components import MACE

from deepchem.models.losses import Loss

class MACELoss(Loss):
    """Loss function for MACE"""

    def __init__(self, energy_weight=1.0):
        super().__init__()
        self.energy_weight = energy_weight

    def _create_pytorch_loss(self):
        """Required by DeepChem - return a PyTorch loss function"""
        # Return nn.MSELoss directly
        return torch.nn.MSELoss()

print("✅ MACELoss defined (using PyTorch MSELoss)")

from deepchem.models import TorchModel
import numpy as np

class MACEWrapper(nn.Module):
    """Wrapper that unpacks PyG batches for MACE"""

    def __init__(self, mace_net):
        super().__init__()
        self.mace_net = mace_net

    def forward(self, inputs):
        """Unpack PyG batch and call MACE"""
        pyg_batch = inputs[0]
        energy, forces = self.mace_net(
            z=pyg_batch.z,
            pos=pyg_batch.pos,
            edge_index=pyg_batch.edge_index,
            batch=pyg_batch.batch
        )
        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
        return energy


class MACEModel(TorchModel):
    """MACE integrated with DeepChem - COMPLETE FIX"""

    def __init__(
        self,
        num_elements=100,
        hidden_dim=64,
        num_interactions=2,
        learning_rate=0.001,
        **kwargs
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        kwargs['device'] = device

        print(f"🔥 Using device: {device}")

        mace_net = MACE(
            num_elements=num_elements,
            hidden_dim=hidden_dim,
            num_interactions=num_interactions
        )

        wrapper = MACEWrapper(mace_net)

        # SET batch_size in kwargs
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = 32

        self._internal_batch_size = kwargs['batch_size']

        super().__init__(
            model=wrapper,
            loss=MACELoss(),
            output_types=['prediction'],
            n_tasks=1,
            learning_rate=learning_rate,
            **kwargs
        )

    def _prepare_batch(self, batch):
        """Convert DeepChem batch to PyG format"""
        if len(batch) == 4:
            inputs, labels, weights, ids = batch
        elif len(batch) == 3:
            inputs, labels, weights = batch
        elif len(batch) == 2:
            inputs, labels = batch
            weights = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        if labels is None:
            y = None
        else:
            y = labels[0] if isinstance(labels, list) else labels
            if hasattr(y, 'shape') and len(y.shape) > 1:
                y = y[:, 0]

        X = inputs[0] if isinstance(inputs, list) else inputs
        device = next(self.model.parameters()).device

        data_list = []
        for i in range(len(X)):
            graph = X[i]
            num_atoms = graph.get_num_atoms()
            atom_features = graph.get_atom_features()

            z = torch.tensor(atom_features[:, 0], dtype=torch.long, device=device)
            pos = torch.randn(num_atoms, 3, device=device)
            pos.requires_grad = True

            adj_list = graph.get_adjacency_list()
            edge_index = []
            for atom_idx, neighbors in enumerate(adj_list):
                for neighbor_idx in neighbors:
                    edge_index.append([atom_idx, neighbor_idx])

            if not edge_index:
                for j in range(num_atoms):
                    for k in range(num_atoms):
                        if j != k:
                            edge_index.append([j, k])

            edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()

            if y is not None:
                y_val = torch.tensor([y[i]], dtype=torch.float32, device=device)
            else:
                y_val = torch.zeros(1, dtype=torch.float32, device=device)

            data = Data(z=z, pos=pos, edge_index=edge_index, y=y_val)
            data_list.append(data)

        pyg_batch = Batch.from_data_list(data_list)

        if weights is None:
            weights = [torch.ones(len(X), dtype=torch.float32, device=device)]
        elif isinstance(weights, np.ndarray):
            weights = [torch.tensor(weights, dtype=torch.float32, device=device)]
        elif not isinstance(weights[0], torch.Tensor):
            weights = [torch.tensor(weights[0], dtype=torch.float32, device=device)]
        else:
            weights = [weights[0].to(device)]

        return ([pyg_batch], [pyg_batch.y], weights)

    def predict(self, dataset, transformers=[], output_types=None):
        """Override predict to handle batching correctly"""
        all_predictions = []

        # Process in batches
        for batch in dataset.iterbatches(batch_size=self._internal_batch_size, deterministic=True):
            X_batch = batch[0]

            self.model.eval()
            with torch.no_grad():
                inputs, _, _ = self._prepare_batch((X_batch, None, None))
                outputs = self.model(inputs)

                # Move to CPU
                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().detach().numpy()

                all_predictions.append(outputs)

        # Concatenate all batches
        result = np.concatenate(all_predictions)
        return result

print("✅ MACEModel with predict override!")

