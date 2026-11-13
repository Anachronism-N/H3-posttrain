from types import SimpleNamespace
from choreobench.providers import NativeProvider


def test_moss_task_prompt_native_template():
    provider=object.__new__(NativeProvider);provider.family='moss';provider.task_prompt='Compose a dance.'
    provider.adapter=SimpleNamespace(processor=SimpleNamespace(_build_default_prompt=lambda text,has_audio:(text,has_audio)))
    assert provider._task_template()==('Compose a dance.',True)


def test_music_flamingo_task_prompt_is_consumed():
    provider=object.__new__(NativeProvider);provider.family='music_flamingo';provider.task_prompt='Compose a dance.'
    provider.adapter=SimpleNamespace(processor=SimpleNamespace(apply_chat_template=lambda messages,**kw:messages))
    message=provider._task_template()[0]
    assert message['content'][0]['type']=='audio'
    assert message['content'][1]['text']=='Compose a dance.'
