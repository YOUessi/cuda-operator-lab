import torch


def test_cuda_environment() -> None:
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability()
    assert (major, minor) == (8, 9)
