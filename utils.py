import numpy as np
import pandas as pd
import torch


def loadData(filename):
    df = pd.read_excel(
        filename,
        header=0,
        index_col=None,
    )
    sensors = list(filter(lambda x: x.endswith(".Pv"), df.columns))
    data = df[sensors].values
    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)
    scaledData = (data - min_values) / (
        max_values - min_values + 1e-6
    )  # 1e-6 is added to avoid division by zero
    return torch.tensor(max_values), torch.tensor(min_values), torch.tensor(scaledData)


def createIndices(data, contextLength):
    indices = torch.arange(0, len(data) - contextLength, contextLength)
    return indices


def getDataloader(filename, contextLength):
    max_values, min_values, scaledData = loadData(filename)
    indices = createIndices(scaledData, contextLength=contextLength)
    indices = indices[torch.randperm(len(indices))]
    return max_values, min_values, scaledData, indices


def saveData(filename, contextLength, train=0.7, val=0.1, test=0.2):
    assert train + val + test == 1
    max_values, min_values, scaledData, indices = getDataloader(filename, contextLength)
    torch.save(
        {
            "max_values": max_values,
            "min_values": min_values,
            "scaledData": scaledData,
            "indices": {
                "train": indices[: int(train * len(indices))],
                "val": indices[
                    int(train * len(indices)) : int((train + val) * len(indices))
                ],
                "test": indices[int((train + val) * len(indices)) :],
            },
            "contextLength": contextLength,
            "missingIndex": {
                "train": torch.concat(
                    [
                        torch.randperm(contextLength).view((1, -1))
                        for _ in range(int(train * len(indices)))
                    ],
                    dim=0,
                ),
                "val": torch.concat(
                    [
                        torch.randperm(contextLength).view((1, -1))
                        for _ in range(
                            int((train + val) * len(indices))
                            - int(train * len(indices))
                        )
                    ],
                    dim=0,
                ),
                "test": torch.concat(
                    [
                        torch.randperm(contextLength).view((1, -1))
                        for _ in range(len(indices) - int((train + val) * len(indices)))
                    ],
                    dim=0,
                ),
            },
            "missingNode": {
                "train": torch.randint(
                    0, scaledData.size(1), (int(train * len(indices)),)
                ),
                "val": torch.randint(
                    0,
                    scaledData.size(1),
                    (int((train + val) * len(indices)) - int(train * len(indices)),),
                ),
                "test": torch.randint(
                    0,
                    scaledData.size(1),
                    (len(indices) - int((train + val) * len(indices)),),
                ),
            },
        },
        filename.replace(".xlsx", ".pt"),
    )
