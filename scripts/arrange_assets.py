"""Index actual downloaded assets; do not manufacture missing templates."""
import json
from pathlib import Path
from backend.storage import atomic_json


def find_one(root, patterns, required=True):
    matches = []
    for pattern in patterns:
        matches.extend(root.rglob(pattern))
    matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p)))
    if not matches:
        if required:
            raise FileNotFoundError(f"模型资产缺失: {patterns}")
        return None
    return matches[0].resolve()


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing asset link has a different target: {destination}")
    elif not destination.exists():
        destination.symlink_to(source, target_is_directory=source.is_dir())


def arrange(root):
    models = root / "models"
    prior = find_one(models / "lhmpp-prior", ["human_model_files"])
    index = {"human_models": str(prior),
             "birefnet": str(find_one(models, ["BiRefNet-general-epoch_244.pth", "birefnet-general-epoch_244.pth"])),
             "flame_map": str(find_one(prior, ["SMPL-X__FLAME_vertex_ids.npy"])),
             "a2e_weights": str(find_one(models / "a2e", ["pretrained_models/lam_audio2exp_streaming.tar"])),
             "gaga_models": str(find_one(models, ["gagatracker"])),
             "arkit_basis": str(find_one(models / "lhmpp-prior", ["flame_arkit_bs.npy"])),
             "flame_model": str(find_one(prior/'flame/2019', ["generic_model.pkl"]))}
    destination = root / "third_party/lhmpp/pretrained_models"
    for name in ("human_model_files", "voxel_grid", "dense_sample_points"):
        source = find_one(models / "lhmpp-prior", [name])
        link(source, destination / name)
    link(Path(index["birefnet"]), destination / Path(index["birefnet"]).name)
    link(find_one(models/'lhmpp-prior',['arcface_resnet18.pth']),destination/'arcface_resnet18.pth')
    gaga=root/'third_party/gaga_track'
    for name in ('emica','flame','matting','vgghead','mask2former'):
        path=Path(index['gaga_models'])/name
        if path.exists(): link(path,gaga/'assets'/name)
    gpt = root / "third_party/gpt_sovits/GPT_SoVITS/pretrained_models"
    for source in (models / "gpt-sovits").iterdir():
        if source.name != ".cache":
            link(source.resolve(), gpt / source.name)
    g2pw=find_one(models/'voice-text',['G2PWModel'],required=False)
    if g2pw: link(g2pw,root/'third_party/gpt_sovits/GPT_SoVITS/text/G2PWModel')
    nltk=find_one(models/'voice-text',['nltk_data'],required=False)
    if nltk: link(nltk,models/'nltk_data')
    atomic_json(models / "assets-index.json", index)
    return index


if __name__ == "__main__":
    print(json.dumps(arrange(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
