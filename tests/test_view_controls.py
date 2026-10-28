import json
from backend.transport.session import Session


def test_portrait_changes_target_and_full_view_restores_it():
    session=Session.__new__(Session)
    session.views={'full':{'yaw':0.,'pitch':0.,'distance':3.,'target':[0.,0.,0.]},
                   'portrait':{'yaw':0.,'pitch':0.,'distance':.8,'target':[0.,1.,0.]}}
    session.camera=dict(session.views['full']);session.channel=None
    session.message(json.dumps({'type':'view','view':'portrait'}))
    assert session.camera['target']==[0.,1.,0.]
    session.message(json.dumps({'type':'camera','yaw':25,'distance':.4}))
    assert session.camera['distance']==.4 and session.camera['target']==[0.,1.,0.]
    assert session.views['portrait']['yaw']==0.
    session.message(json.dumps({'type':'view','view':'full'}))
    assert session.camera==session.views['full']


def test_camera_does_not_zoom_past_the_native_head_detail_range():
    session=Session.__new__(Session)
    session.channel=None;session.camera={'yaw':0.,'pitch':0.,'distance':3.}
    session.camera_limits={'min_distance':1.2}
    session.message(json.dumps({'type':'camera','distance':.4}))
    assert session.camera['distance']==1.2
