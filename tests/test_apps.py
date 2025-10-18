from apps.gradio.app import app

def test_gradio():
    output = app('Test')
    assert output == 'Test'
