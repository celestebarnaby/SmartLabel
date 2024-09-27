class TimeOutException(Exception):
    def __init__(self, message, errors=None):
        super(TimeOutException, self).__init__(message)
        self.errors = errors


def handler(signum, frame):
    raise TimeOutException("Timeout")
