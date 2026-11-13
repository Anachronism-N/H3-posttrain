from captionlab.review_bundle import _same_segment_provider

def identity(path):
    return dict(model='unchanged',artifact_identity=dict(inference_adapter_source=dict(path=path,sha256='a'*64,size_bytes=10)))

def test_copied_identical_adapter_source():
    assert _same_segment_provider(identity('/old/code.py'),identity('/new/code.py'))

def test_changed_model_adapter_or_unknown_identity_refused():
    a=identity('/a');b=identity('/b');b['artifact_identity']['inference_adapter_source']['sha256']='b'*64
    assert not _same_segment_provider(a,b)
    b=identity('/b');b['model']='different'
    assert not _same_segment_provider(a,b)
