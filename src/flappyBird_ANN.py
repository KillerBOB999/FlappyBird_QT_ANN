import neuralNetwork
from neuralNetwork import NeuralNetwork

import edge
from edge import Edge

def main(playerX: int, playerY: int, pipeInfo: list, stillAlive: int): 
    test1 = Edge(0, 3, 100, 100)
    test2 = Edge(1, 3, 100, 100)
    test3 = Edge(2, 3, 100, 100)

    supertest = (test1, test2, test3)
    NNTest = NeuralNetwork(supertest, 4, (3))


    if pipeInfo[0]['x'] >= 0:
        print("Pipe Info = ", pipeInfo[0]['x'])
    else:
        print("Pipe Info = ", pipeInfo[1]['x'])
    return

main(1, 2, 3, 1)