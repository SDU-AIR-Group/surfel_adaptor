import importlib

__attributes = {
    
    'TerrainFeat2Render': 'terrain_feat2render',
    
    'SparseStructureLatent': 'sparse_structure_latent',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    
    from .terrain_feat2render import TerrainFeat2Render
    
    from .sparse_structure_latent import (
        SparseStructureLatent,
        TextConditionedSparseStructureLatent,
        ImageConditionedSparseStructureLatent,
    )    