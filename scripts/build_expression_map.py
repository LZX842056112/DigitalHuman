from pathlib import Path
import ast
import numpy as np

from backend.avatar.expressions import ExpressionMap
from backend.storage import read_json, atomic_json


def build_expression_map(root: Path, folder: Path, renderer):
    import torch
    assets = read_json(root / "models/assets-index.json")
    arkit = np.load(assets["arkit_basis"], allow_pickle=False)
    if isinstance(arkit, np.lib.npyio.NpzFile):
        raise ValueError("ARKit 资产必须提供明确的52维位移数组，不能猜测NPZ字段")
    if arkit.ndim != 3 or arkit.shape[0] != 52 or arkit.shape[2] != 3:
        raise ValueError(f"ARKit 基形状无效: {arkit.shape}")
    correspondence = np.load(assets["flame_map"], allow_pickle=False).reshape(-1)
    if len(correspondence) != arkit.shape[1]:
        raise ValueError("SMPL-X/FLAME共同顶点映射不完整")
    layer = renderer.model.renderer.smplx_model.smplx_layer
    # Fit the exact FLAME directions already installed in this renderer. This
    # prevents a different FLAME release from silently changing the basis.
    flame_basis=layer.expr_dirs.detach().cpu().numpy()[correspondence]
    if flame_basis.shape!=(arkit.shape[1],3,100):
        raise ValueError('渲染器表情基与 ARKit 顶点对应关系不一致')
    parameters = renderer.state["parameters"]
    def vertices(jaw):
        with torch.no_grad():
            out = layer(betas=parameters["betas"], expression=torch.zeros(1,100,device="cuda"),
                        body_pose=torch.zeros(1,63,device="cuda"), global_orient=torch.zeros(1,3,device="cuda"),
                        jaw_pose=torch.as_tensor(jaw,device="cuda",dtype=torch.float32).reshape(1,3),
                        left_hand_pose=torch.zeros(1,45,device="cuda"),right_hand_pose=torch.zeros(1,45,device="cuda"),
                        leye_pose=torch.zeros(1,3,device="cuda"),reye_pose=torch.zeros(1,3,device="cuda"))
        return out.vertices[0,correspondence].cpu().numpy()
    neutral = vertices([0,0,0])
    jacobian = []
    for axis in range(3):
        angle = np.zeros(3, np.float32)
        angle[axis] = .01
        jacobian.append((vertices(angle) - neutral) / .01)
    jaw_basis = np.stack(jacobian,axis=-1).reshape(-1,3)
    mapping = ExpressionMap.fit(flame_basis.reshape(-1,100),
                                arkit.transpose(1,2,0).reshape(-1,52),jaw_basis, ridge=1e-5)
    mapping.save(folder / "expression_map.npz")
    tree=ast.parse((root/'third_party/a2e/models/utils.py').read_text())
    channels=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='ARKitBlendShape' for target in node.targets):
            channels=ast.literal_eval(node.value)
    if not isinstance(channels,list) or len(channels)!=52 or len(set(channels))!=52:
        raise ValueError('上游未提供完整的 ARKit 通道名称')
    atomic_json(folder / "expression_map_report.json", {"arkit_dimensions":52,"flame_dimensions":100,
                "vertices":len(correspondence),"jaw":"finite-difference Jacobian, radians",
                "arkit_asset":assets["arkit_basis"],"flame_asset":assets["flame_model"],
                'channels':channels,'basis_source':'actual renderer FLAME expression directions'})
    return mapping
