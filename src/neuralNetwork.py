import edge
import math

class NeuralNetwork(object):
    """description of class"""
    _edgeDict = dict(dict())
    _nextID = int()
    _outLayerIDs = list()

    def __init__(self, edges: list(), nextID: int, outLayerIDs: list()):
        self._nextID = nextID
        self._outLayerIDs = outLayerIDs
        self.makeEdgeDict(edges)
        return

    def makeEdgeDict(self, edges: list()):
        for synapse in edges:
            if synapse._inNeuronID not in self._edgeDict:
                self._edgeDict[synapse._inNeuronID] = { synapse._outNeuronID: synapse }
            else:
                self._edgeDict[synapse._inNeuronID][synapse._outNeuronID] = synapse
        return

    def feedForward(self, inputValues: list()):
        numOfInputs = len(inputValues)
        currentProgress = dict()
        output = dict()
        for inNeuron in self._edgeDict.keys():
            if inNeuron not in currentProgress:
                currentProgress[inNeuron] = inputValues[inNeuron]
            for outNeuron in self._edgeDict[inNeuron].keys():
                if outNeuron not in currentProgress:
                    currentProgress[outNeuron] = currentProgress[inNeuron] * self._edgeDict[inNeuron][outNeuron]._weight
                else:
                    currentProgress[outNeuron] += currentProgress[inNeuron] * self._edgeDict[inNeuron][outNeuron]._weight
                if outNeuron in self._outLayerIDs:
                    output[outNeuron] = currentProgress[outNeuron]
        return output

    def sigmoid(self, numeratorFactor: float, exponentfactor: float, inputValue: float):
        return numeratorFactor / (1 + math.exp(-exponentfactor * inputValue))