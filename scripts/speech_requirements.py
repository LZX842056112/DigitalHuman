"""Keep the GPT environment on CPU and retain its required transformer API."""
from pathlib import Path


def normalize(text):
    result=[]
    for line in text.splitlines():
        if line.strip().startswith('onnxruntime'):
            if 'onnxruntime==1.19.2' not in result: result.append('onnxruntime==1.19.2')
        else:
            result.append(line)
    return '\n'.join(result)+'\n'


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    text=(root/'third_party/gpt_sovits/requirements.txt').read_text()
    (root/'.runtime/speech-requirements.txt').write_text(normalize(text))
