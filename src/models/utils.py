# utils.py: Utility functions for models
def save_model(model, path):
    import torch
    torch.save(model.state_dict(), path)

def load_model(model, path):
    import torch
    model.load_state_dict(torch.load(path))

