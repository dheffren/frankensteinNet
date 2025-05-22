import torch
def get_fixed_batch(dataloader, seed=42, num_samples=64):
    """
    Selects a fixed batch of samples from a dataloader deterministically.
    Should return the same batch every time this is called with same seed.

    Args:
        dataloader: the DataLoader object (e.g., val_loader)
        seed: seed for reproducible selection
        num_samples: how many examples to collect

    Returns:
        A batch (x, y) or just x depending on the dataset.
    """
    dataset = dataloader.dataset
    g = torch.Generator().manual_seed(seed)

    indices = torch.randperm(len(dataset), generator=g)[:num_samples]
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=num_samples,
        shuffle=False,  # very important
        num_workers=0
    )

    return next(iter(loader))