from apps.gradio.app import app

def test_gradio(monkeypatch):
    def fake_model(x):
        return x
    monkeypatch.setattr('apps.gradio.app.model', fake_model)
    output = app('Test')
    assert output == 'Test'
