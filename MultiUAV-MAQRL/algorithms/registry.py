"""Extensible specification lookup and adapter-based algorithm construction."""
from .maddpg import MADDPGSpec

_FACTORIES = {"maddpg": MADDPGSpec}


def register_algorithm(name, factory):
    if name in _FACTORIES:
        raise ValueError(f"Algorithm already registered: {name}")
    if not name or not callable(factory):
        raise ValueError("Require a name and callable factory")
    _FACTORIES[name] = factory


def available_algorithms():
    return tuple(sorted(_FACTORIES))


def create_algorithm(name, *, adapter=None, config=None):
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(f"Unknown algorithm: {name}") from None
    if adapter is not None:
        if name != 'maddpg':
            return factory(adapter=adapter, config=config)
        from .maddpg.maddpg import MADDPG
        return MADDPG(adapter.agent_ids,adapter.global_state_dim,config,adapter.observation_dim)
    return factory()


def load_algorithm(name, checkpoint, device='cpu'):
    """Load through an algorithm-specific loader behind a shared interface."""
    if name not in _FACTORIES:
        raise ValueError(f'Unknown algorithm: {name}')
    if name == 'maddpg':
        from .maddpg.maddpg import MADDPG
        return MADDPG.load_checkpoint(checkpoint,device=device)
    loader=getattr(_FACTORIES[name],'load_checkpoint',None)
    if loader is None: raise ValueError(f'No checkpoint loader registered for {name}')
    return loader(checkpoint,device=device)
