class Debug():
    def __init__(self):
        self.show_reformed_batch_size = False,
    def set(self, key, value):
        self.dict[key] = value
    def __call__(self, relating_parameters, *args):
        for p in relating_parameters:
            if not p: return
        print(*args)

debug = Debug()