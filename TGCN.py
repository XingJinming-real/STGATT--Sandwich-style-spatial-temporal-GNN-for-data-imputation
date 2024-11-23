# ----------------------------------------
# **TGCN.py** is a file that contains the implementation of the Temporal Graph Convolutional Network (TGCN) model.
# ----------------------------------------

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, inputSize, hiddenSize, numLayers):
        super().__init__()
        self.gcn = gnn.GCN(
            inputSize, hiddenSize, numLayers, add_self_loops=False, dropout=0.1
        )

    def forward(self, x, edgeIndex):
        # x is nodeNum x timeLen x featureNum
        x = self.gcn(x, edgeIndex)
        return x


class GRU(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super().__init__()
        self.gru = nn.GRUCell(inputSize, hiddenSize)

    def forward(self, x, hx):
        # x is Batch x Time x featureNum
        x = self.gru(x, hx)
        return x


class TGCN(nn.Module):
    def __init__(self, inputSize, numLayers):
        super().__init__()
        self.inProject = nn.Linear(1, inputSize)
        self.simLinear = nn.Linear(inputSize, inputSize)
        self.gcn = GCN(inputSize, inputSize, numLayers)
        self.gru = GRU(inputSize, inputSize)
        self.out = nn.Linear(inputSize, 1)
        self.sparsity = 0.5

    def createGraph(self, x):
        # **createGraph** is a function that creates a graph based on the similarity of the input data.
        # * It's the same as the one in TGATT.
        # x: Batch x Time x Features, Note B is the number of sensors
        x = self.simLinear(x)
        x1 = x.unsqueeze(0)  # 1 x Batch x Time x Features
        x2 = x.unsqueeze(1)  # Batch x 1 x Time x Features
        x2t = x2.transpose(2, 3)  # Batch x 1 x Features x Time
        relation = torch.matmul(x1, x2t)  # Batch x Batch x Time x Time
        relationMatrix = torch.mean(
            torch.diagonal(relation, dim1=2, dim2=3), dim=-1
        )  # Batch x Batch

        relationMatrix[torch.eye(relationMatrix.size(0), dtype=torch.bool)] = (
            0  # exclude self loops
        )
        thresh = torch.quantile(relationMatrix, 1 - self.sparsity)

        edge_index = (
            torch.concat(torch.where(relationMatrix > thresh), dim=0)
            .view((2, -1))
            .type(torch.long)
        )
        return edge_index

    def forward(self, x):
        # x is Batch x Time x 1
        x = self.inProject(x)  # Batch x Time x inputSize
        hx = torch.zeros(
            x.size(0), x.size(2)
        )  # Batch x outputSize, where inputSize == outputSize
        edgeIndex = self.createGraph(x)
        xNew = []
        for i in range(x.size(1)):
            xNew.append(self.gcn(x[:, i, :], edgeIndex))
        xNew = F.relu(torch.stack(xNew, dim=1))  # Batch x Time x inputSize
        output = []
        for i in range(x.size(1)):
            hx = self.gru(xNew[:, i, :], hx)
            output.append(hx)
        output = F.relu(torch.stack(output, dim=1))  # Batch x Time x inputSize
        output = self.out(output)  # Batch x Time x 1
        return output
