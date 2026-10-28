"""Batched fixed-topology kinematics with capture-safe quaternion selection."""
import torch
import torch.nn.functional as F

_topologies=[]
_indices={}


def _topology(parents):
    for tensor,version,values in _topologies:
        if tensor is parents and version==parents._version: return values
    values=tuple(parents.detach().cpu().tolist())
    if any(not 0<=parent<index for index,parent in enumerate(values) if index):
        raise ValueError('骨架父关节顺序无效')
    _topologies.append((parents,parents._version,values))
    if len(_topologies)>8: _topologies.pop(0)
    return values


def batch_rigid_transform(rot_mats,joints,parents,dtype=torch.float32):
    topology=_topology(parents)
    key=(topology,str(joints.device))
    if key not in _indices:
        depth=[0]*len(topology)
        for index,parent in enumerate(topology):
            if index: depth[index]=depth[parent]+1
        groups=[]
        for level in range(1,max(depth)+1):
            children=[i for i,d in enumerate(depth) if d==level]
            groups.append((torch.tensor(children,device=joints.device),
                           torch.tensor([topology[i] for i in children],device=joints.device)))
        _indices[key]=(torch.tensor(topology[1:],device=joints.device),groups)
    parent_indices,groups=_indices[key]
    relative=torch.cat([joints[:,:1],joints[:,1:]-joints.index_select(1,parent_indices)],1)
    upper=torch.cat([rot_mats,relative[...,None]],-1)
    bottom=F.pad(torch.zeros_like(upper[:,:,:1,:3]),[0,1],value=1)
    local=torch.cat([upper,bottom],-2)
    transforms=torch.zeros_like(local)
    transforms[:,0]=local[:,0]
    for children,parent in groups:
        values=transforms.index_select(1,parent)@local.index_select(1,children)
        transforms.index_copy_(1,children,values)
    homogeneous=F.pad(joints[...,None],[0,0,0,1])
    return transforms[:,:,:3,3],transforms-F.pad(transforms@homogeneous,[3,0,0,0,0,0])


def matrix_to_quaternion(matrix):
    # The four diagonal candidates are the standard robust matrix-to-quaternion
    # construction. A fixed-size gather replaces data-dependent boolean indexing.
    m00,m01,m02=matrix[...,0,0],matrix[...,0,1],matrix[...,0,2]
    m10,m11,m12=matrix[...,1,0],matrix[...,1,1],matrix[...,1,2]
    m20,m21,m22=matrix[...,2,0],matrix[...,2,1],matrix[...,2,2]
    values=torch.stack([1+m00+m11+m22,1+m00-m11-m22,1-m00+m11-m22,1-m00-m11+m22],-1)
    positive=values>0
    roots=torch.sqrt(torch.where(positive,values,torch.ones_like(values)))
    absolute=torch.where(positive,roots,torch.zeros_like(roots))
    candidates=torch.stack([
        torch.stack([absolute[...,0].square(),m21-m12,m02-m20,m10-m01],-1),
        torch.stack([m21-m12,absolute[...,1].square(),m10+m01,m02+m20],-1),
        torch.stack([m02-m20,m10+m01,absolute[...,2].square(),m12+m21],-1),
        torch.stack([m10-m01,m20+m02,m21+m12,absolute[...,3].square()],-1)],-2)
    candidates=candidates/(2*absolute.clamp_min(.1)[...,None])
    index=absolute.argmax(-1)[...,None,None].expand(*absolute.shape[:-1],1,4)
    quaternion=candidates.gather(-2,index).squeeze(-2)
    return torch.where(quaternion[...,:1]<0,-quaternion,quaternion)


def axis_angle_to_matrix(axis_angle):
    angle=torch.linalg.vector_norm(axis_angle,dim=-1,keepdim=True)
    half=angle*.5
    vector=axis_angle*(.5*torch.sinc(half/torch.pi))
    w=torch.cos(half)[...,0];x,y,z=vector.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(*axis_angle.shape[:-1],3,3)
