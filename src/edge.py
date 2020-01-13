class Edge(object):
    """description of class"""
    _inNeuronID = None
    _outNeuronID = None
    _weight = None
    _bias = None

    def __init__(self, inNeuronID: int, outNeuronID: int, weight: float, bias: float):
        self._inNeuronID = inNeuronID
        self._outNeuronID = outNeuronID
        self._weight = weight
        self._bias = bias
        return