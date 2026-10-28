"""ModelScope publishes the A2E checkpoint with an uppercase LAM prefix."""
import platform
import tempfile
import unittest
from pathlib import Path

from scripts.arrange_assets import arrange


@unittest.skipUnless(platform.system() == 'Linux', 'Checks Linux case-sensitive asset paths and links')
class AssetCaseTest(unittest.TestCase):
    def test_arrange_uses_extracted_a2e_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                'lhmpp-prior/human_model_files/flame/2019/generic_model.pkl',
                'lhmpp-prior/human_model_files/SMPL-X__FLAME_vertex_ids.npy',
                'lhmpp-prior/BiRefNet-general-epoch_244.pth',
                'lhmpp-prior/flame_arkit_bs.npy',
                'lhmpp-prior/arcface_resnet18.pth',
                'a2e/LAM_audio2exp_streaming.tar',
                'a2e/pretrained_models/lam_audio2exp_streaming.tar',
            ]
            for name in files:
                target = root/'models'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            for name in ['lhmpp-prior/gagatracker', 'lhmpp-prior/voxel_grid',
                         'lhmpp-prior/dense_sample_points', 'gpt-sovits']:
                (root/'models'/name).mkdir(parents=True, exist_ok=True)
            index = arrange(root)
            self.assertEqual(Path(index['a2e_weights']), root/'models/a2e/pretrained_models/lam_audio2exp_streaming.tar')
            self.assertTrue((root/'models/assets-index.json').is_file())
