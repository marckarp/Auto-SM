from auto_model import AutoModel

print("AutoModel: ", AutoModel)

class VertexModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
class DatabricksModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class DataikuModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class KubernetesModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)