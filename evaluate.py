import torch
from dataset import ModelNet40, ModelLoNet40
from dataloader import get_dataloader
from config import Config
from models import SAO
import train
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

config = Config()

root = config.root
train_cls = config.train_cls
val_cls = config.val_cls
test_cls = config.test_cls


def test(use_A=False, use_B=False, test_set_name="test"):
    train_set = ModelNet40(root=root, mode="train", cls_=[train_cls, val_cls], use_B=use_B)
    val_set = ModelNet40(root=root, mode="val", cls_=[train_cls, val_cls], use_B=use_B)
    test_set = ModelNet40(root=root, mode="test", cls_=test_cls, use_B=use_B)
    test_lo_set = ModelLoNet40(test_set)

    train_set.config, val_set.config, test_set.config, test_lo_set.config = config, config, config, config
    train_loader, neighborhood_limits = get_dataloader(
        train_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True
    )
    val_loader, _ = get_dataloader(
        val_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        neighborhood_limits=neighborhood_limits
    )
    test_loader, _ = get_dataloader(
        test_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        neighborhood_limits=neighborhood_limits
    )
    test_lo_loader, _ = get_dataloader(
        test_lo_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        neighborhood_limits=neighborhood_limits
    )

    params_path = {
        "0_0": "./params/sao-no-A-no-B.pth",
        "0_1": "./params/sao-no-A-only-B.pth",
        "1_0": "./params/sao-no-B-only-A.pth",
        "1_1": "./params/sao-A-and-B.pth"
    }
    model = SAO(config, use_A=use_A, use_B=use_B)
    model.to(config.device)
    model.load_state_dict(torch.load(params_path[str(int(use_A)) + "_" + str(int(use_B))]))
    loader = None
    if test_set_name == "train":
        loader = train_loader
    elif test_set_name == "val":
        loader = val_loader
    elif test_set_name == "test":
        loader = test_loader
    else:
        loader = test_lo_loader
    train.evaluate(model, loader)


if __name__ == '__main__':
    # test(use_A=False, use_B=False, test_set_name="test_lo")
    # test(use_A=True, use_B=False, test_set_name="test_lo")
    test(use_A=False, use_B=True, test_set_name="test_lo")
    # test(use_A=True, use_B=True, test_set_name="test_lo")
    # test(use_A=True, use_B=True, test_set_name="test")