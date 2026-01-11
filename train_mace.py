print("="*70)
print("🔥 DAY 1 FINAL - Y KO BHI FLATTEN KARO")
print("="*70)

import deepchem as dc
import numpy as np
import time

# Load
print("\n1. Loading QM9...")
tasks, datasets, _ = dc.molnet.load_qm9(featurizer='GraphConv', splitter='random', reload=True)
train_full, valid_full, _ = datasets


train_y_flat = train_full.y[:100, 0]
valid_y_flat = valid_full.y[:20, 0]

train_w = train_full.w[:100, 0:1] if len(train_full.w.shape) == 2 else train_full.w[:100].reshape(-1, 1)
valid_w = valid_full.w[:20, 0:1] if len(valid_full.w.shape) == 2 else valid_full.w[:20].reshape(-1, 1)

print(f"   Train y shape: {train_y_flat.shape}")
print(f"   Valid y shape: {valid_y_flat.shape}")

train_energy = dc.data.NumpyDataset(
    X=train_full.X[:100],
    y=train_y_flat,
    w=train_w,
    ids=train_full.ids[:100]
)
valid_energy = dc.data.NumpyDataset(
    X=valid_full.X[:20],
    y=valid_y_flat,
    w=valid_w,
    ids=valid_full.ids[:20]
)

# Model
print("\n2. Creating MACE...")
mace_model = MACEModel(hidden_dim=32, num_interactions=2, batch_size=10)

# Train
print("\n3. Training...")
start = time.time()
mace_model.fit(train_energy, nb_epoch=10)
print(f"   ✅ {time.time()-start:.1f}s")

# Evaluate
print("\n4. Evaluating...")
metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
score = mace_model.evaluate(valid_energy, [metric])

print(f"\n📊 MAE: {score['mean_absolute_error']:.3f} kcal/mol")
print("\n🎉🎉🎉 DAY 1 COMPLETE! 🎉🎉🎉")
print("="*70)