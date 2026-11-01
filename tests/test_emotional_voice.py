def test_voice_prompt_cache_separates_voice_identity_and_emotion():
    from backend.voice.engine import voice_cache_key
    neutral=voice_cache_key('voice1','neutral','model1')
    assert neutral == voice_cache_key('voice1','neutral','model1')
    assert neutral != voice_cache_key('voice1','happy','model1')
    assert neutral != voice_cache_key('voice2','neutral','model1')
    assert neutral != voice_cache_key('voice1','neutral','model2')


def test_inference_configuration_does_not_import_training_data_processors():
    from backend.voice.engine import inference_yaml
    import yaml
    source='sample_rate: 24000\nfilter: !name:cosyvoice.dataset.processor.filter {}\ndata_pipeline: [!ref "<filter>"]\ntrain_conf: {max_epoch: 200}\n'
    assert yaml.safe_load(inference_yaml(source)) == {'sample_rate':24000}
def test_text_frontend_uses_verified_local_assets_and_restores_download_function(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    import pytest
    from backend.voice import engine
    def network(*args,**kwargs):raise AssertionError('Inference attempted network access')
    wetext=SimpleNamespace(snapshot_download=network)
    monkeypatch.setitem(sys.modules,'wetext',SimpleNamespace(wetext=wetext))
    monkeypatch.setattr(engine,'verified_model',lambda root,key:tmp_path/key)
    with engine.local_text_frontend(tmp_path):
        assert wetext.snapshot_download('pengzhendong/wetext') == str(tmp_path/'text_normalizer')
        with pytest.raises(ValueError):wetext.snapshot_download('different/model')
    assert wetext.snapshot_download is network
