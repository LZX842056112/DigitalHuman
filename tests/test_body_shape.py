import numpy as np
import pytest
import torch
from backend.avatar import lhmpp
from backend.errors import PipelineError


def test_body_shape_uses_raw_prediction_without_unused_translation_view():
    class Estimator:
        def __call__(self, image, is_shape=True):
            location=torch.zeros(2,3).T
            if is_shape:
                location.view(1,1,-1)  # Same noncontiguous layout as the real upstream failure.
            return {'loc':location,'dist':torch.ones(1),'shape':torch.arange(10).float()[None]}
    betas=lhmpp.fit_body_shape(Estimator(),np.zeros((32,32,3),np.uint8))
    np.testing.assert_array_equal(betas,np.arange(10,dtype=np.float32))


@pytest.mark.parametrize('result',[{}, {'shape':torch.full((1,10),float('nan'))}])
def test_body_shape_rejects_missing_or_nonfinite_fit(result):
    with pytest.raises(PipelineError):
        lhmpp.fit_body_shape(lambda *args,**kwargs:result,np.zeros((32,32,3),np.uint8))
