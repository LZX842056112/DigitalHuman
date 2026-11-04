"""Shared service commands must preserve launch and migration contracts."""
import json
from pathlib import Path

import pytest

from scripts import services


def full_plan(root, monkeypatch):
    model = root/'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'fixture model')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'is_file', lambda _: True)
        return services.service_plan(root)


def test_shared_service_commands_keep_original_arguments_and_order(tmp_path, monkeypatch):
    plan = full_plan(tmp_path, monkeypatch)
    assert [item['name'] for item in plan] == ['dialogue', 'voice', 'portrait', 'studio']
    commands = [
        ['/opt/digitalhuman/llama.cpp/build/bin/llama-server', '-m',
         str(tmp_path/'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf'), '--host', '127.0.0.1',
         '--port', '8081', '-ngl', '0', '-t', '6', '-c', '4096', '--alias', 'qwen2.5-3b-instruct'],
        ['/opt/digitalhuman/emotional-voice/bin/python', '-m', 'uvicorn',
         'backend.voice.service:app', '--host', '127.0.0.1', '--port', '9880'],
        ['/opt/digitalhuman/portrait/bin/python', '-m', 'uvicorn',
         'backend.portrait.service:app', '--host', '127.0.0.1', '--port', '9881'],
        ['/opt/digitalhuman/portrait/bin/python', '-m', 'uvicorn',
         'backend.api.app:app', '--host', '127.0.0.1', '--port', '7860'],
    ]
    assert plan == [dict(name=name, port=port, cwd=tmp_path, command=command, health=health)
        for name, port, command, health in zip(
            ['dialogue', 'voice', 'portrait', 'studio'], [8081, 9880, 9881, 7860],
            commands, ['/health', '/health', '/health', '/api/health'])]


def test_studio_only_plan_needs_no_model_or_executable_checks(tmp_path, monkeypatch):
    def unexpected(*args):
        pytest.fail('studio-only checked a model or executable')

    monkeypatch.setattr(Path, 'glob', unexpected)
    monkeypatch.setattr(Path, 'is_file', unexpected)
    plan = services.service_plan(tmp_path, studio_only=True)
    assert [item['name'] for item in plan] == ['studio']
    assert plan[0]['port'] == 7860 and plan[0]['health'] == '/api/health'


@pytest.mark.parametrize('stale', [None, 'voice', 'portrait', 'dialogue'])
def test_studio_only_checks_running_workers_with_the_same_commands(tmp_path, monkeypatch, stale):
    plan = full_plan(tmp_path, monkeypatch)
    records = {item['name']: {'pid': 1, 'command': item['command'], 'port': item['port']}
               for item in plan}
    if stale is not None:
        records[stale]['command'] = [*records[stale]['command'], '--obsolete']
    folder = tmp_path/'.runtime'
    folder.mkdir()
    registry = folder/'services.json'
    registry.write_text(json.dumps(records))
    monkeypatch.setattr(services, 'alive', lambda record: record.get('pid') == 1)
    monkeypatch.setattr(services, '_check_health', lambda item,timeout=2: None)
    monkeypatch.setattr(services.subprocess, 'Popen', lambda *a, **k: pytest.fail('launched a process'))
    monkeypatch.setattr(Path, 'glob', lambda *a: pytest.fail('checked LLM model availability'))
    monkeypatch.setattr(Path, 'is_file', lambda *a: pytest.fail('checked worker executable availability'))
    if stale is None:
        services.start(tmp_path, studio_only=True)
    else:
        with pytest.raises(RuntimeError, match=stale):
            services.start(tmp_path, studio_only=True)
    assert json.loads(registry.read_text()) == records
