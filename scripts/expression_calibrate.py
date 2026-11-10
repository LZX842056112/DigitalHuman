"""Render the four expression presets for one avatar and report the differences.

The presets only change the Ditto emotion condition and the expression gain, so
a correct result keeps the frame count, the background and the face identity
while moving the mouth and brow. This script turns that into numbers instead of
an opinion. It needs the GPU worker environment and a prepared avatar.

    python scripts/expression_calibrate.py --avatar <id>
    python scripts/expression_calibrate.py --avatar <id> --audio data/avatars/<id>/voice/sample.wav
"""
import argparse
import json
import os
from pathlib import Path

import av
import numpy as np

ROOT=Path(os.environ.get('DIGITAL_HUMAN_ROOT') or Path(__file__).resolve().parents[1]).resolve()
PRESETS=('natural','smile','serious','surprise')


def read_frames(path):
    frames=[]
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
    return frames


def compare_frames(base, other, outside=None, border=.05):
    """Compare one rendered preset against the neutral render.

    ``outside`` marks pixels the paste-back mask cannot touch (alpha == 0).
    When it is given, that is the background test; the coarse border check
    stays for callers that do not have the mask.
    """
    base=np.asarray(base,np.int16);other=np.asarray(other,np.int16)
    if base.shape!=other.shape:raise ValueError('两段视频尺寸不一致')
    diff=np.abs(base-other).max(axis=2)
    changed=diff>2
    height,width=diff.shape[:2]
    if changed.any():
        rows,cols=np.nonzero(changed)
        bbox=[int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())]
    else:
        bbox=None
    margin=max(1,int(min(height,width)*border))
    border_same=bool(np.array_equal(
        base[:margin].astype(np.uint8),other[:margin].astype(np.uint8)) and
        np.array_equal(base[-margin:].astype(np.uint8),other[-margin:].astype(np.uint8)) and
        np.array_equal(base[:,:margin].astype(np.uint8),other[:,:margin].astype(np.uint8)) and
        np.array_equal(base[:,-margin:].astype(np.uint8),other[:,-margin:].astype(np.uint8)))
    values=np.abs(base.astype(np.float32)-other.astype(np.float32))
    result={'changed_ratio':float(changed.mean()),
            'mean_abs_diff':float(values[changed].mean()) if changed.any() else 0.,
            'bbox':bbox,'border_identical':border_same}
    if outside is not None:
        # A pixel with alpha == 0 must keep the photo value. Allow +-2 for
        # uint8 rounding at the very edge of the feathered mask.
        delta=np.abs(base[outside].astype(np.int16)-other[outside].astype(np.int16))
        result['outside_max_delta']=int(delta.max()) if delta.size else 0
        result['outside_identical']=bool(result['outside_max_delta']<=2)
    return result


def mask_outside(cache_folder,width,height):
    """Pixels the compositor cannot repaint, derived from the cached affine."""
    import sys
    import cv2
    sys.path.insert(0,str(ROOT/'third_party/ditto'))
    from core.utils.get_mask import get_mask
    state=next(Path(cache_folder).glob('*.npz'))
    with np.load(state,allow_pickle=False) as data:
        matrix=data['M_c2o'] if 'M_c2o' in data.files else data['M_c2o_lst'][0]
    alpha=cv2.warpAffine(get_mask(512,512,.9,.9),matrix[:2],(width,height))
    if alpha.ndim==3:alpha=alpha[...,0]
    return alpha<=1e-6


def summarize(name,base_frames,other_frames,outside=None):
    if len(base_frames)!=len(other_frames):
        return {'preset':name,'frames':len(other_frames),'passed':False,
                'reason':'帧数与 natural 不一致'}
    worst={'changed_ratio':0.,'mean_abs_diff':0.,'border_identical':True,
            'outside_identical':True,'outside_max_delta':0}
    for base,other in zip(base_frames,other_frames):
        result=compare_frames(base,other,outside)
        worst['changed_ratio']=max(worst['changed_ratio'],result['changed_ratio'])
        worst['mean_abs_diff']=max(worst['mean_abs_diff'],result['mean_abs_diff'])
        worst['border_identical']=worst['border_identical'] and result['border_identical']
        worst['outside_identical']=worst['outside_identical'] and result.get('outside_identical',True)
        worst['outside_max_delta']=max(worst['outside_max_delta'],result.get('outside_max_delta',0))
    stable=worst['outside_identical'] if outside is not None else worst['border_identical']
    passed=(stable and worst['changed_ratio']<.5 and worst['mean_abs_diff']>=1.)
    return {'preset':name,'frames':len(other_frames),**worst,'passed':passed,
            'reason':'' if passed else '掩膜外像素变化或差异过小/过大'}


def main():
    parser=argparse.ArgumentParser(description='Compare expression presets on a real avatar')
    parser.add_argument('--avatar',required=True)
    parser.add_argument('--audio',default='')
    parser.add_argument('--out',default='')
    options=parser.parse_args()

    from backend.photo_profiles import active_variant,migrate_profile,source_path
    from backend.portrait.engine import PhotoEngine
    folder=ROOT/'data/avatars'/options.avatar
    manifest=migrate_profile(json.loads((folder/'avatar.json').read_text(encoding='utf-8')))
    variant=active_variant(manifest)
    if not variant or not variant.get('portrait'):
        raise SystemExit('该人物还没有准备造型')
    cache=source_path(folder,variant['portrait']['folder'])
    audio=Path(options.audio) if options.audio else folder/'voice/sample.wav'
    if not audio.is_file():raise SystemExit('缺少参考音频：'+str(audio))
    output=Path(options.out) if options.out else ROOT/'outputs/expression'/options.avatar
    output.mkdir(parents=True,exist_ok=True)

    engine=PhotoEngine(ROOT)
    rendered={}
    for preset in PRESETS:
        target=output/f'{preset}.mkv'
        print(f'渲染 {preset} -> {target}')
        engine.render(cache,audio,target,'neutral',expression=preset)
        rendered[preset]=read_frames(target)

    base=rendered['natural']
    outside=mask_outside(cache,variant['portrait']['width'],variant['portrait']['height'])
    rows=[summarize(preset,base,rendered[preset],outside)
          for preset in PRESETS if preset!='natural']
    report={'avatar':options.avatar,'variant':variant['id'],
            'frames':len(base),'audio':str(audio),'presets':rows,
            'note':'数值是初值，需人工复核；serious 与 natural 的可辨性是重点'}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

    print(f'帧数 {len(base)}')
    for row in rows:
        icon='PASS' if row['passed'] else 'FAIL'
        print(f"{icon} {row['preset']:<10} 变化像素 {row['changed_ratio']*100:5.1f}%  "
              f"平均差 {row['mean_abs_diff']:5.2f}  掩膜外一致 {row['outside_identical']}")
    print('报告：',output/'report.json')
    if not all(row['passed'] for row in rows):raise SystemExit(1)


if __name__=='__main__':
    main()
