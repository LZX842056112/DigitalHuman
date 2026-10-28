import numpy as np


def test_tracking_keeps_original_frame_identity_when_a_detection_is_missing():
    from scripts.track_faces import pack_observations
    frames={'face_00002':{'tracking_frame':2,'file':'000120.jpg'},
            'face_00006':{'tracking_frame':6,'file':'000360.jpg'}}
    optimized={'face_00006':{name:np.zeros(n) for name,n in [('expcode',100),('posecode',6),('eyecode',6),('shapecode',300)]}}
    base={'face_00006':{'vgg_results':{'lmks_2d70':np.ones((70,2))*123}}}
    result=pack_observations(optimized,base,frames)
    assert list(result)==['6']
    assert result['6']['file']=='000360.jpg'
    assert np.asarray(result['6']['observed_lmks68']).shape==(68,2)


def test_tracking_fingerprint_changes_with_exact_crop_selection_or_model(tmp_path):
    from scripts.track_faces import tracking_input_manifest
    from backend.storage import atomic_json
    root=tmp_path;folder=root/'person';faces=folder/'processed/faces'
    faces.mkdir(parents=True);(root/'models').mkdir()
    atomic_json(root/'models/lock.json',{'tracker':'version1'})
    selected=[{'tracking_frame':2,'file':'two.jpg'}]
    (faces/'two.jpg').write_bytes(b'exact source crop')
    first=tracking_input_manifest(root,folder,selected)
    assert first==tracking_input_manifest(root,folder,selected)
    (faces/'three.jpg').write_bytes(b'new photo')
    more=selected+[{'tracking_frame':1000000,'file':'three.jpg'}]
    assert first!=tracking_input_manifest(root,folder,more)
    (faces/'two.jpg').write_bytes(b'changed source crop')
    assert first!=tracking_input_manifest(root,folder,selected)
    crop_changed=tracking_input_manifest(root,folder,selected)
    atomic_json(root/'models/lock.json',{'tracker':'version2'})
    assert crop_changed!=tracking_input_manifest(root,folder,selected)
    previous=tracking_input_manifest(root,folder,selected)
    (root/'configs').mkdir();atomic_json(root/'configs/models.json',{'gaga_track':{'commit':'new source'}})
    assert previous!=tracking_input_manifest(root,folder,selected)


def test_legacy_tracking_is_not_reused_after_photo_import(tmp_path):
    from scripts.track_faces import tracking_current,tracking_input_manifest
    from backend.storage import atomic_json
    folder=tmp_path/'person';faces=folder/'processed/faces';faces.mkdir(parents=True)
    (faces/'one.jpg').write_bytes(b'original')
    selected=[{'tracking_frame':1,'file':'one.jpg'}]
    atomic_json(folder/'processed/tracking/flame.json',{'1':{}})
    assert not tracking_current(tmp_path,folder,selected)
    atomic_json(folder/'processed/tracking/input_manifest.json',tracking_input_manifest(tmp_path,folder,selected))
    assert tracking_current(tmp_path,folder,selected)
    (faces/'two.jpg').write_bytes(b'new portrait')
    assert not tracking_current(tmp_path,folder,selected+[{'tracking_frame':1000000,'file':'two.jpg'}])
