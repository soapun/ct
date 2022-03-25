from dataclasses import dataclass
import typing
import numpy as np

@dataclass
class Func:
    
    func : typing.Union[str, int]
    def __post_init__(self):
        func_text = f'''def _qrfunction(self, t):\n    return {self.func}'''
        exec(func_text)
        self.func = locals()['_qrfunction']

@dataclass
class ConnectionInfo:
    id : int
    square : float
    lmbda : float

@dataclass
class PartInfo:
    id : int
    name : str
    square : float
    
    c : float
    eps : float
    Q_R : Func
    
    connection_info : list[ConnectionInfo]
    
converters = {
    Func : lambda s: Func(s)
}

@dataclass
class Part:
    scene : object
    scene_scale : object
    scene_trans : object
    
    info : PartInfo = None