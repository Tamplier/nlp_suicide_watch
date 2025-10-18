import sys
from unittest.mock import MagicMock

# Don't load any models in test env
fake_model_module = MagicMock()
fake_model_module.predict = lambda x: x
sys.modules['src.scripts.model_load'] = fake_model_module
