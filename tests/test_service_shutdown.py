"""A stop request must retain ownership until the service actually exits."""
import json
import pytest
from scripts import services


def registry(root):
    folder = root / '.runtime'
    folder.mkdir()
    record = {'pid': 1234, 'start_ticks': 'fixture-start', 'port': 9880}
    path = folder / 'services.json'
    path.write_text(json.dumps({'voice': record}))
    return path, record


@pytest.mark.parametrize('state, expected', [('S', '123'), ('Z', None), ('X', None)])
def test_exited_process_cannot_keep_service_alive(monkeypatch, state, expected):
    fields = [state, *['0'] * 18, '123']
    monkeypatch.setattr(services.Path, 'read_text', lambda *args, **kwargs: '42 (worker) ' + ' '.join(fields))
    assert services.identity(42) == expected


def test_stop_waits_for_owned_process_before_removing_record(tmp_path, monkeypatch):
    path, record = registry(tmp_path)
    exited = False
    requests, waits = [], []
    monkeypatch.setattr(services, 'alive', lambda item: not exited)
    monkeypatch.setattr(services.os, 'killpg', lambda pid, sig: requests.append(pid), raising=False)
    monkeypatch.setattr(services.socket.socket, 'connect_ex', lambda *args: 0 if not exited else 1)

    def finish(_):
        nonlocal exited
        # The ownership file must survive until process/port shutdown completes.
        assert json.loads(path.read_text())['voice'] == record
        waits.append(True)
        exited = True

    monkeypatch.setattr(services.time, 'sleep', finish)
    services.stop(tmp_path, ['voice'])
    assert requests == [record['pid']]
    assert waits, 'stop returned before the service could exit'
    assert json.loads(path.read_text()) == {}


def test_stop_timeout_keeps_ownership_for_retry(tmp_path, monkeypatch):
    path, record = registry(tmp_path)
    monkeypatch.setattr(services, 'alive', lambda item: True)
    monkeypatch.setattr(services.os, 'killpg', lambda *args: None, raising=False)
    monkeypatch.setattr(services.socket.socket, 'connect_ex', lambda *args: 0)
    clock = iter([0, 60, 120, 180])
    monkeypatch.setattr(services.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(services.time, 'sleep', lambda _: None)
    with pytest.raises(TimeoutError, match='voice'):
        services.stop(tmp_path, ['voice'])
    assert json.loads(path.read_text())['voice'] == record


def test_stop_waits_for_listener_after_owned_pid_exits(tmp_path, monkeypatch):
    path, record = registry(tmp_path)
    exited = released = False
    monkeypatch.setattr(services, 'alive', lambda item: not exited)

    def terminate(*_):
        nonlocal exited
        exited = True

    def release(_):
        nonlocal released
        assert json.loads(path.read_text())['voice'] == record
        released = True

    monkeypatch.setattr(services.os, 'killpg', terminate, raising=False)
    monkeypatch.setattr(services.socket.socket, 'connect_ex', lambda *args: 1 if released else 0)
    monkeypatch.setattr(services.time, 'sleep', release)
    services.stop(tmp_path, ['voice'])
    assert released
    assert json.loads(path.read_text()) == {}


def test_stop_never_signals_reused_foreign_pid(tmp_path, monkeypatch):
    path, _ = registry(tmp_path)
    monkeypatch.setattr(services, 'alive', lambda item: False)
    monkeypatch.setattr(services.os, 'killpg', lambda *args: pytest.fail('signalled a foreign process'), raising=False)
    services.stop(tmp_path, ['voice'])
    assert json.loads(path.read_text()) == {}
