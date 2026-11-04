"""Backend preparation stages must have Chinese labels in the active UI."""
import ast
from pathlib import Path
import re


def test_every_emitted_preparation_stage_has_a_ui_label():
    root=Path(__file__).resolve().parents[1]
    stages=set()
    for relative in ('backend/api/app.py','backend/storage.py'):
        for node in ast.walk(ast.parse((root/relative).read_text(encoding='utf-8'))):
            if isinstance(node,ast.keyword) and node.arg=='stage' and isinstance(node.value,ast.Constant):
                stages.add(node.value.value)
            if isinstance(node,ast.Dict):
                for key,value in zip(node.keys,node.values):
                    if (isinstance(key,ast.Constant) and key.value=='stage'
                            and isinstance(value,ast.Constant)):stages.add(value.value)
    source=(root/'frontend/src/photo-main.tsx').read_text(encoding='utf-8')
    labels=re.search(r'const stageName[^=]*=\{([^}]+)\}',source).group(1)
    names=set(re.findall(r'(\w+)\s*:',labels))
    assert stages and stages<=names,f'Backend stages without UI labels: {sorted(stages-names)}'
