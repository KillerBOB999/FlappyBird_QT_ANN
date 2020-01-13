import edge
import math

class NeuralNetwork(object):
    """description of class"""
    _edgeDict = dict(dict())
    _nextID = None
    _outLayerIDs = list()

    def __init__(self, edges: list(), nextID: int, outLayerIDs: list()):
        self._nextID = nextID
        self._outLayerIDs = outLayerIDs
        self.makeEdgeDict(edges)
        return

    def makeEdgeDict(self, edges: list()):
        self._edgeDict.clear()
        for edge in edges:
            if edge._inNeuronID not in self._edgeDict:
                self._edgeDict[edge._inNeuronID] = { edge._outNeuronID: edge }
        return

    def feedForward(self, inputValues: list()):
        numOfInputs = len(inputValues)
        currentprogress = dict()
        currentprogress.clear()
        for inNeuron in self._edgeDict:
            for outNeuron in self._edgeDict:
                # TODO
                print()
        return

    def sigmoid(self, numeratorFactor: float, exponentfactor: float, inputValue: float):
        return numeratorFactor / (1 + math.exp(-exponentfactor * inputValue))