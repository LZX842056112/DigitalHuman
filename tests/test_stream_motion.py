import math

import numpy as np


def test_hubert_windows_are_identical_across_audio_chunk_boundaries():
    from backend.portrait.stream import FeatureStream
    pcm=np.random.default_rng(6).normal(0,.1,65537).astype(np.float32)
    def features(parts):
        windows=[]
        def infer(window):
            windows.append(window.copy())
            return np.full((5,1024),len(windows),np.float32)
        stream=FeatureStream(infer)
        values=[stream.push(part) for part in parts];values.append(stream.finish())
        return np.concatenate(values),windows
    whole,windows=features([pcm]);split,split_windows=features(np.array_split(pcm,31))
    np.testing.assert_array_equal(whole,split)
    assert len(whole)==math.ceil(len(pcm)/640)
    for first,second in zip(windows,split_windows):np.testing.assert_array_equal(first,second)


def test_motion_stream_publishes_stable_prefix_and_flushes_exact_frame_count():
    from backend.portrait.stream import MotionStream
    class Motion:
        seq_frames=80;valid_clip_len=70;overlap_v2=10
        def __call__(self,values,sequence):
            block=values.copy()
            return block if sequence is None else np.concatenate([sequence[:,:-10],block],axis=1)
        def _smo(self,sequence,start,end):return sequence.copy()
        def cvt_fmt(self,values):return [row.copy() for row in values[0]]
    stream=MotionStream(Motion(),lambda values,index:values)
    first=stream.push(np.arange(100,dtype=np.float32)[:,None])
    assert 0<len(first)<100
    second=stream.push(np.arange(100,161,dtype=np.float32)[:,None])
    last=stream.finish()
    actual=np.asarray(first+second+last).reshape(-1)
    np.testing.assert_array_equal(actual,np.arange(161,dtype=np.float32))


def test_final_smoothing_does_not_mutate_future_motion_context():
    from backend.portrait.stream import MotionStream
    class Motion:
        seq_frames=80;valid_clip_len=70;overlap_v2=10
        def __call__(self,values,sequence):
            return values.copy() if sequence is None else np.concatenate([sequence[:,:-10],values],axis=1)
        def _smo(self,sequence,start,end):
            original=sequence.copy()
            for index in range(start,end):
                sequence[:,index]=np.mean(original[:,max(0,index-1):index+2],axis=1)
            return sequence
        def cvt_fmt(self,values):return [row.copy() for row in values[0]]
    values=np.random.default_rng(44).normal(size=(231,1)).astype(np.float32)
    def generate(parts):
        stream=MotionStream(Motion(),lambda block,index:block);out=[]
        for part in parts:out.extend(stream.push(part))
        out.extend(stream.finish());return np.asarray(out)
    expected=generate([values])
    np.testing.assert_array_equal(generate(np.array_split(values,21)),expected)
