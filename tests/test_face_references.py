import json
import numpy as np
import pytest


def test_photo_face_references_are_idempotent_and_preserve_original(tmp_path, monkeypatch):
    import cv2
    from backend.pipeline.media import append_face_photos
    folder=tmp_path/'person'
    (folder/'source').mkdir(parents=True)
    (folder/'processed/faces').mkdir(parents=True)
    image=np.full((100,80,3),150,np.uint8)
    cv2.imwrite(str(folder/'source/face.jpg'),image)
    original=(folder/'source/face.jpg').read_bytes()
    (folder/'processed/faces/frames.json').write_text(json.dumps({'selected':[],'all':[]}))
    class Detector:
        def detectMultiScale(self,*args,**kwargs): return np.array([[20,20,40,40]])
    monkeypatch.setattr(cv2,'CascadeClassifier',lambda *_:Detector())
    append_face_photos(folder,['face.jpg'])
    append_face_photos(folder,['face.jpg'])
    frames=json.loads((folder/'processed/faces/frames.json').read_text())['selected']
    assert len(frames)==1 and frames[0]['source_photo']=='face.jpg'
    assert frames[0]['tracking_frame']>=1000000
    assert (folder/'source/face.jpg').read_bytes()==original


@pytest.mark.parametrize('boxes,accepted',[
    ([[70,160,10,10],[30,10,50,50]],True),
    ([[10,20,35,35],[55,20,32,32]],False),
])
def test_photo_rejects_ambiguous_faces_but_keeps_a_dominant_face(tmp_path,monkeypatch,boxes,accepted):
    import cv2
    from backend.pipeline.media import append_face_photos
    folder=tmp_path/'person';faces=folder/'processed/faces'
    faces.mkdir(parents=True);(folder/'source').mkdir()
    image=np.full((200,100,3),150,np.uint8)
    cv2.imwrite(str(folder/'source/face.jpg'),image)
    (faces/'frames.json').write_text(json.dumps({'selected':[],'all':[]}))
    class Detector:
        def detectMultiScale(self,*args,**kwargs): return np.array(boxes)
    monkeypatch.setattr(cv2,'CascadeClassifier',lambda *_:Detector())
    append_face_photos(folder,['face.jpg'])
    selected=json.loads((faces/'frames.json').read_text())['selected']
    assert bool(selected)==accepted
    if accepted:
        assert selected[0]['detected_bbox']==[30,10,50,50]
        assert selected[0]['bbox']==[13,0,97,77]
        assert selected[0]['detection_count']==2


def test_new_photo_order_preserves_tracking_ids(tmp_path,monkeypatch):
    import cv2
    from backend.pipeline.media import append_face_photos
    faces=tmp_path/'processed/faces';faces.mkdir(parents=True)
    (tmp_path/'source').mkdir()
    cv2.imwrite(str(tmp_path/'source/a.jpg'),np.ones((100,100,3),np.uint8)*100)
    existing={'source_photo':'b.jpg','tracking_frame':1000001,'file':'photo_001.jpg'}
    (faces/'frames.json').write_text(json.dumps({'selected':[existing],'all':[existing]}))
    class Detector:
        def detectMultiScale(self,*args,**kwargs): return np.array([[20,20,40,40]])
    monkeypatch.setattr(cv2,'CascadeClassifier',lambda *_:Detector())
    append_face_photos(tmp_path,['a.jpg','b.jpg'])
    selected=json.loads((faces/'frames.json').read_text())['selected']
    assert [item['tracking_frame'] for item in selected]==[1000000,1000001]
    assert selected[-1]==existing
