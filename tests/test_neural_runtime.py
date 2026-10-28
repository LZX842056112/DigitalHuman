from types import SimpleNamespace
import torch
from backend.avatar.runtime_cache import RuntimeCache


def cache_for_test(enabled=True):
    cache=RuntimeCache.__new__(RuntimeCache)
    cache.renderer=SimpleNamespace(_runtime_render=True,model=SimpleNamespace(neural_renderer=torch.nn.Linear(3,3)))
    cache.compiled=None;cache.compile_error=None;cache.stats={}
    cache.compile_enabled=enabled
    return cache


def test_compiler_is_used_only_for_runtime_and_keeps_calibration_gradients(monkeypatch):
    cache=cache_for_test();calls=[]
    def compiler(module,**options):
        calls.append(options)
        return module
    monkeypatch.setattr(torch,'compile',compiler)
    value=torch.ones(1,3,requires_grad=True)
    cache.neural(value).sum().backward()
    assert not calls and torch.isfinite(value.grad).all()
    with torch.no_grad():
        expected=cache.renderer.model.neural_renderer(value)
        torch.testing.assert_close(cache.neural(value),expected)
        torch.testing.assert_close(cache.neural(value*2),cache.renderer.model.neural_renderer(value*2))
    assert len(calls)==1 and calls[0]['fullgraph'] is True
    assert cache.stats['neural_compiled'] is True
    cache.renderer._runtime_render=False
    value.grad=None
    cache.neural(value).sum().backward()
    assert len(calls)==1 and torch.isfinite(value.grad).all()


def test_compilation_failure_is_recorded_and_uses_original_network(monkeypatch):
    cache=cache_for_test();attempts=[]
    def unavailable(*args,**kwargs):
        attempts.append(1)
        raise RuntimeError('compiler unavailable')
    monkeypatch.setattr(torch,'compile',unavailable)
    value=torch.ones(1,3)
    with torch.no_grad():
        for _ in range(2):
            torch.testing.assert_close(cache.neural(value),cache.renderer.model.neural_renderer(value))
    assert len(attempts)==1
    assert cache.stats['neural_compiled'] is False
    assert 'unavailable' in cache.stats['neural_compile_error']


def test_compiler_factory_does_not_inherit_render_autocast_or_no_grad(monkeypatch):
    cache=cache_for_test();contexts=[]
    def compiler(module,**options):
        contexts.append((torch.is_grad_enabled(),torch.is_autocast_enabled()))
        return module
    monkeypatch.setattr(torch,'compile',compiler)
    with torch.no_grad(),torch.autocast('cuda'):
        cache.neural(torch.ones(1,3))
    assert contexts==[(True,False)]


def test_disabled_compiler_uses_verified_eager_network(monkeypatch):
    cache=cache_for_test(enabled=False)
    def unexpected(*args,**kwargs): raise AssertionError('disabled compiler invoked')
    monkeypatch.setattr(torch,'compile',unexpected)
    value=torch.ones(1,3)
    with torch.no_grad():
        torch.testing.assert_close(cache.neural(value),cache.renderer.model.neural_renderer(value))
