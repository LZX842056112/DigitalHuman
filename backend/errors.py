class NotReady(RuntimeError):
    """The requested real model or asset has not passed preparation."""


class PipelineError(RuntimeError):
    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(message)
