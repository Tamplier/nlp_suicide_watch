import inspect

class PickleCompatible:
    def __setstate__(self, state):
        self.__dict__.update(state)
        sig = inspect.signature(self.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if not hasattr(self, param_name) and param.default != inspect.Parameter.empty:
                setattr(self, param_name, param.default)
