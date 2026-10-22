import torch

from bpm.model import CNNBiLSTM


def test_forward_shape_matches_batch_size():
    model = CNNBiLSTM()
    x = torch.randn(4, 1, 1000)

    out = model(x)

    assert out.shape == (4,)
