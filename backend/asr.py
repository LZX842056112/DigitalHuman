"""Offline speech recognition for the voice reference transcript.

The transcript is what zero-shot cloning needs: CosyVoice aligns the reference
speech tokens with the matching words, so a real transcript keeps the speaker's
rhythm instead of letting the model invent one. Recognition runs on CPU with the
same local faster-whisper model the audio input path already uses.
"""
import threading
from pathlib import Path

_MODELS={}
_LOCK=threading.Lock()


def model_folder(root, folder='models/asr/small'):
    path=Path(root)/folder
    if not (path/'model.bin').is_file():
        raise RuntimeError('语音识别模型尚未安装，请运行 setup.ps1')
    return path


def recognizer(root, folder='models/asr/small'):
    """Load one shared CPU recognizer; repeated calls reuse the weights."""
    path=model_folder(root,folder).resolve()
    with _LOCK:
        instance=_MODELS.get(str(path))
        if instance is None:
            from faster_whisper import WhisperModel
            instance=WhisperModel(str(path),device='cpu',compute_type='int8',
                                  cpu_threads=4,local_files_only=True)
            _MODELS[str(path)]=instance
        return instance


def transcribe(root, audio, *, folder='models/asr/small', language=None):
    """Return (text, detected language) for one short clip."""
    segments,info=recognizer(root,folder).transcribe(str(audio),language=language,
        vad_filter=True,beam_size=3,condition_on_previous_text=False,temperature=0.)
    text=''.join(segment.text for segment in segments).strip()
    return ' '.join(text.split()),getattr(info,'language',None)


def usable(text, minimum=4):
    """A prompt transcript is only worth using when it covers real words."""
    return isinstance(text,str) and len(text.strip())>=minimum
