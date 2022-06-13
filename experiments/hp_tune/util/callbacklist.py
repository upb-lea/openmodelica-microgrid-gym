class CallbackList(list):
    # List of callback functions
    def fire(self, *args, **kwargs):
        # executes all callbacks in list
        for listener in self:
            listener(*args, **kwargs)
