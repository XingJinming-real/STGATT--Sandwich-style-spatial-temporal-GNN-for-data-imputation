import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch_geometric.nn as gnn
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from utils import saveData
from tqdm import tqdm
from baselines import FFN, BGRU, TransformerEncoder
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from TGCN import TGCN


class Transformer(nn.Module):
    def __init__(
        self,
        input_dimm,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers,
        contextLength=16,
    ):
        # * Note, contextLength is a hyperparameter that is used to create the positional encoding
        super().__init__()
        self.encoderLayer = nn.TransformerEncoderLayer(
            d_model=input_dimm,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.net = nn.TransformerEncoder(self.encoderLayer, num_layers=num_layers)
        self.out = nn.Linear(input_dimm, output_dim)
        self.relu = nn.ReLU()
        self.pe = nn.Parameter(
            torch.zeros(
                1, contextLength, input_dimm, requires_grad=True, dtype=torch.float
            )
        )  # 1 x Time x Features

    def forward(self, x):
        # x: Batch x Time x Features
        x = x + self.pe
        x = self.net(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super().__init__()
        self.net = gnn.GAT(
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            heads=num_heads,
            num_layers=num_layers,
            add_self_loops=False,
        )

    def forward(self, x, edge_index):
        # x: Batch x Time x Features, Note B is the number of sensors (nodes)
        processedX = []
        for i in range(x.size(1)):
            processedX.append(self.net(x[:, i, :], edge_index))
        return torch.stack(processedX, dim=1)


class MyNet(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_heads, num_layers, sparsity=0.5
    ):
        super().__init__()
        self.sparsity = sparsity
        self.inLinear = nn.Linear(1, input_dim)
        self.gat = GAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.transformer1 = Transformer(
            input_dimm=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.transformer2 = Transformer(
            input_dimm=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        self.outLienar = nn.Linear(output_dim, 1)
        self.simLinear = nn.Linear(input_dim, input_dim)

    def createGraph(self, x):
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
        # Batch x Time x Features, Note B is the number of sensors
        x = self.inLinear(x)
        edge_index = self.createGraph(x)
        x = x + self.transformer1(x)
        x = x + self.gat(x, edge_index)
        x = x + self.transformer2(x)
        x = self.outLienar(x)
        return x


def train(
    net,
    info,
    epoch,
    missingRate,
    device="cuda",
    patience=5,
    filename="data/22June2020.pt",
):
    missingNum = int(info["contextLength"] * missingRate)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    bestEvalLoss = float("inf")
    bestEvalParams = None
    patienceCounter = 0
    for i in range(epoch):
        if patienceCounter >= patience:
            break
        epochLoss = 0
        for j, indice in tqdm(
            enumerate(info["indices"]["train"]),
            total=len(info["indices"]["train"]),
            desc="Training",
        ):
            optimizer.zero_grad()
            missingNodeIndex = info["missingNode"]["train"][j]
            predictionIndex = info["missingIndex"]["train"][
                j, :missingNum
            ]  # get the missing index, like [3,5,10,1] for each sample
            x = (
                info["scaledData"][indice : indice + info["contextLength"]]
                .t()
                .unsqueeze(dim=-1)
                .type(torch.float)
            ).to(
                device
            )  # x is Batch x Time x 1
            y = x[missingNodeIndex, predictionIndex].detach().clone()
            x[missingNodeIndex, predictionIndex] = 0
            out = net(x)  # out is Batch x Time x 1
            loss = criterion(out[missingNodeIndex, predictionIndex], y)
            loss.backward()
            optimizer.step()
            epochLoss += loss.item()
        print("Epoch: ", i, "Loss: ", epochLoss)
        patienceCounter += 1
        # ------------------- Eval -------------------
        mse, mae, rmse, mape = test(net, info, "val", missingRate)
        if mse <= bestEvalLoss:
            bestEvalLoss = mse
            bestEvalParams = net.state_dict()
            patienceCounter = 0

        # -------------------------------------------
    torch.save(
        bestEvalParams,
        f"checkpoint/{filename.strip('.xlsx pt').split('/')[-1]}_{net.__class__.__name__}_bestEvalParams.pt",
    )

    # ------------------- Runnable Test -------------------
    # x = torch.randn(5, 10, 1)  # 5 sensors, 10 time steps, 1 feature
    # optimizer.zero_grad()
    # out = net(x)
    # loss = criterion(out, torch.randn(5, 10, 1))
    # loss.backward()
    # optimizer.step()
    # print(loss.item())
    # ------------------------------------------------------


@torch.no_grad()
def test(net, info, dataName, missingRate):
    missingNum = int(info["contextLength"] * missingRate)
    net.eval()
    data = info["scaledData"]
    indices = info["indices"][dataName]
    mae, rmse, mape = 0, 0, 0
    mse = 0
    for i in range(len(indices)):
        indice = indices[i]
        x = (
            data[indice : indice + info["contextLength"]]
            .t()
            .unsqueeze(dim=-1)
            .type(torch.float)
        )  # x is Batch x Time x 1
        missingNodeIndex = info["missingNode"][dataName][i]
        predictionIndex = info["missingIndex"][dataName][i, :missingNum]
        y = x[missingNodeIndex, predictionIndex].detach().clone()
        x[missingNodeIndex, predictionIndex] = 0
        out = net(x)
        out = out[missingNodeIndex, predictionIndex]
        mse += F.mse_loss(out, y).detach().item()
        out = out.detach().cpu().numpy()
        y = y.cpu().numpy()
        mae += mean_absolute_error(y, out)
        rmse += root_mean_squared_error(y, out)
        mape += mean_absolute_percentage_error(y, out)

    mse, mae, rmse, mape = (
        mse / len(indices),
        mae / len(indices),
        rmse / len(indices),
        mape / len(indices),
    )
    print("--" * 20, missingRate, "--" * 20)
    print(
        net.__class__.__name__,
        ": " "MSE",
        mse,
        "MAE: ",
        mae,
        "RMSE: ",
        rmse,
        "MAPE: ",
        mape,
    )
    net.train()
    return mse, mae, rmse, mape


def preprocessing(filename, contextLength):
    saveData(filename, contextLength)


def main(filename, epoch, missingRate=0.5, device="cuda", is_train=True):
    info = torch.load(filename)
    assert info["missingIndex"]["test"].size(0) == info["indices"]["test"].size(0)
    nets = [
        MyNet(input_dim=8, hidden_dim=16, output_dim=8, num_heads=4, num_layers=2),
        FFN(8),
        BGRU(8, 16, 2),
        TransformerEncoder(8, 16, 2, 4),
        TGCN(8, 2),
    ]
    if is_train:
        # ------------------- Train -------------------
        assert not isinstance(missingRate, list) and missingRate == 0.5
        for net in nets:
            net = net.to(device)
            train(net, info, epoch, missingRate, device, filename=filename)
        # -------------------------------------------
    else:
        # ------------------- Test -------------------
        assert isinstance(missingRate, list)
        print("---" * 20, " Testing ", "---" * 20)
        writter = pd.ExcelWriter(
            f"results/{filename.strip('.xlsx pt').split('/')[-1]}.xlsx"
        )
        for mRate in missingRate:
            df = {}
            for net in nets:
                stateDict = torch.load(
                    f"checkpoint/{filename.strip('.xlsx pt').split('/')[-1]}_{net.__class__.__name__}_bestEvalParams.pt"
                )
                net.load_state_dict(stateDict)
                mse, mae, rmse, mape = test(net, info, "test", mRate)
                df[net.__class__.__name__] = [mse, mae, rmse, mape]
            df = pd.DataFrame(df, index=["MSE", "MAE", "RMSE", "MAPE"])
            df.to_excel(writter, sheet_name=str(mRate))
        writter.close()


# preprocessing("data/22June2020.xlsx", 16)
# preprocessing("data/29June2020.xlsx", 16)

# main("data/22June2020.pt", epoch=100, device="cpu", is_train=True, missingRate=0.5)
# main("data/29June2020.pt", epoch=100, device="cpu", is_train=True, missingRate=0.5)
main(
    "data/22June2020.pt",
    epoch=100,
    device="cpu",
    is_train=False,
    missingRate=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
)
main(
    "data/29June2020.pt",
    epoch=100,
    device="cpu",
    is_train=False,
    missingRate=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
)
# ------------------------------------------------------
