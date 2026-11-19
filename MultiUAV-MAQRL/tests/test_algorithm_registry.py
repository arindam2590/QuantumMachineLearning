import pytest
from algorithms.registry import create_algorithm,available_algorithms
from algorithms.cli import prepare


def test_maddpg_registry_is_specification_only():
    spec=create_algorithm('maddpg')
    assert 'maddpg' in available_algorithms()
    assert not spec.implemented and spec.action_dim==3 and spec.observation_dim==157
    with pytest.raises(ValueError): create_algorithm('missing')


@pytest.mark.parametrize('mode',['train','validate','test'])
def test_shared_preparation(mode):
    result=prepare(mode,['--algo','maddpg','--num-drones','4'])
    assert result['mode']==mode and result['num_drones']==4
    assert not result['implemented']


def test_future_factory_and_loader_registration(monkeypatch):
    from algorithms import registry
    # Contract test doubles only; no additional learning algorithm.
    class FutureFactory:
        def __call__(self,**kwargs): return ('created',kwargs)
        def load_checkpoint(self,path,device): return ('loaded',path,device)
    monkeypatch.setitem(registry._FACTORIES,'future_contract_test',FutureFactory())
    assert 'future_contract_test' in registry.available_algorithms()
    assert registry.create_algorithm('future_contract_test',adapter='adapter',config='config')[0]=='created'
    assert registry.load_algorithm('future_contract_test','model.pt','cpu')==('loaded','model.pt','cpu')
