import argparse
import json
import logging
import os
import sys
import time
from os.path import join

import boto3
import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

from PIL import Image

import zipfile
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


img_transform= {
    'dataset':
    transforms.Compose([
        transforms.Resize((28, 28)), 
        transforms.ToTensor(),
    ]),
    'training':
    transforms.Compose([
         transforms.RandomResizedCrop(size=224),
         transforms.RandomHorizontalFlip(),
    ]),
    'validation': 
    transforms.Compose([
        transforms.Resize(size=224),
    ]),
    'testing': 
    transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]),
}


if "SAGEMAKER_METRICS_DIRECTORY" in os.environ:
    log_file_handler = logging.FileHandler(
        join(os.environ["SAGEMAKER_METRICS_DIRECTORY"], "metrics.json")
    )
    formatter = logging.Formatter(
        "{'time':'%(asctime)s', 'name': '%(name)s', \
    'level': '%(levelname)s', 'message': '%(message)s'}",
        style="%",
    )
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        print("x1", x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print("x2", x.shape)
        x = x.view(64, -1)
        print("x3", x.shape)
        x = F.relu(self.fc1(x))
        print("x4", x.shape)
        x = F.dropout(x, training=self.training)
        print("x5", x.shape)
        x = self.fc2(x)
        print("x6", x.shape)
        return F.log_softmax(x, dim=1)

    
    
    
def check_dataset_loaded():
    print("-------------------------")
    print("check_dataset_loaded")
    print("-------------------------")
    
    result = []
    found = False
    cwd = os.getcwd()
    for root, dirr, files in os.walk(cwd):
        if "dataset_reduced.zip" in files:
            found = True
            break
            
    return found
    
def load_dataset():
    print("-------------------------")
    print("load_dataset")
    print("-------------------------")
    
    if (check_dataset_loaded() == False):
        cwd = os.getcwd()
        print("----------------------------")
        print("current dir: ", cwd)
        s3_client = boto3.client("s3")
        bucket = "sagemaker-hackathon-demo-eu-west-1-017233837209"
        s3_client.download_file(
        bucket,
        "hackathon/dataset_reduced.zip",
        "dataset_reduced.zip",
        )
        
        
        with zipfile.ZipFile("./dataset_reduced.zip", 'r') as zip_ref:
            zip_ref.extractall("./")
        
        print("-------------------------")
        print("after zip")
    
    
    
def _get_data_loaders(batch_sizeP, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    
    print("-------------------------")
    print("_get_train_data_loader")
    print("-------------------------")
    
    
#     dataset = datasets.MNIST(
#         training_dir,
#         train=True,
#         transform=transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
#         ),
#         download=False,
#     )
#     train_sampler = (
#         torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
#     )
#     return torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=train_sampler is None,
#         sampler=train_sampler,
#         **kwargs
#     )

    
    
    load_dataset()
    train_dataset = datasets.ImageFolder("./final_dataset/train/", transform=img_transform['dataset'])
    
    print("------------------------------------")
    print('#{} train images loaded succesfully!'.format(len(train_dataset)))
    
    SPLIT_SIZE = .1
    n_val = round(SPLIT_SIZE * len(train_dataset))

    train_set, val_set = torch.utils.data.random_split(train_dataset, [len(train_dataset)-n_val, n_val])

    print('Spliting train images to #{} training samples and #{} samples for validation'.format(len(train_set), len(val_set)))
    
    print("batch_sizeP", batch_sizeP)
    train_setB = DataLoader(train_set, batch_size= batch_sizeP, shuffle=False)
    val_setB = DataLoader(val_set, batch_size= batch_sizeP, shuffle=False)
    
    print("------------------------------")
    print("train_setB", train_setB)
    print("val_setB", val_setB)
    return train_setB, val_setB
    
    

# def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
#     logger.info("Get test data loader")
    
#     print("-------------------------")
#     print("_get_test_data_loader")
#     print("-------------------------")
    
    
#     load_dataset()
#     test_dataset = datasets.ImageFolder("./final_dataset/GT/", transform=img_transform['dataset'])
    
#     print("------------------------------------")
#     print('#{} test images loaded succesfully!'.format(len(test_dataset)))
    
#     test_setB = DataLoader(test_dataset, batch_size= 32, shuffle=False)
#     print("------------------------------")
#     print("test_setB", test_setB)
#     return test_setB


def _average_gradients(model):
    # Gradient averaging.
    
    print("-------------------------")
    print("_average_gradients")
    print("-------------------------")
    
    
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args, tracker=None):
    print("-------------------------")
    print("train")
    print("-------------------------")
    
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} nodes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current host rank is {}. Number of gpus: {}".format(dist.get_rank(), args.num_gpus)
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

#     train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
#     test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    train_loader, val_loader = _get_data_loaders(args.batch_size, args.data_dir, is_distributed, **kwargs)

#     logger.info(
#         "Processes {}/{} ({:.0f}%) of train data".format(
#             len(train_loader.sampler),
#             len(train_loader.dataset),
#             100.0 * len(train_loader.sampler) / len(train_loader.dataset),
#         )
#     )

#     logger.info(
#         "Processes {}/{} ({:.0f}%) of test data".format(
#             len(test_loader.sampler),
#             len(test_loader.dataset),
#             100.0 * len(test_loader.sampler) / len(test_loader.dataset),
#         )
#     )

    model = Net().to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("-------------------------------")
    print("loop")
    args.epochs = 1
    for epoch in range(1, args.epochs + 1):
        print("---------------------")
        print("EPOCH = ", epoch)
        print("train data_loader size = ", len(train_loader))
        it = 0
        
        
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            if data.shape[0] == args.batch_size:
                print("----------------------------")
                print("it = ", it, "of ", len(train_loader))
                it += 1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                print("---------------------")
                print("data.shape", data.shape)
                output = model(data)
                output = output
                print("output.shape", output.shape)
                print("target.shape", target.shape)
                loss = F.nll_loss(output, target)
                loss.backward()
                if is_distributed and not use_cuda:
                    # average gradients manually for multi-machine cpu case only
                    _average_gradients(model)
                optimizer.step()
                if batch_idx % args.log_interval == 0:
                    logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f};".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.sampler),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
                    
            print("Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.6f};".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.sampler),
                100.0 * batch_idx / len(train_loader),
                loss.item(),))
            
            evaluate(model, val_loader, device, tracker)
        save_model(model, args.model_dir)


def evaluate(model, val_loader, device, tracker=None):
    
    print("-------------------------")
    print("evaluate")
    print("-------------------------")
    
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if data.shape[0] == args.batch_size:
                print("data.shape.val", data.shape)
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(val_loader.dataset)
                print(
                    "Test Average loss: {:.4f}, Test Accuracy: {:.0f}%;\n".format(
                    test_loss, 100.0 * correct / len(val_loader.dataset)
                )
                )


def model_fn(model_dir):
    
    print("-------------------------")
    print("model_fn")
    print("-------------------------")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_channels = int(os.environ.get("hidden_channels", "5"))
    kernel_size = int(os.environ.get("kernel_size", "5"))
    dropout = float(os.environ.get("dropout", "0.5"))
    model = torch.nn.DataParallel(Net(hidden_channels, kernel_size, dropout))
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
        return model.to(device)


def save_model(model, model_dir):
    
    print("-------------------------")
    print("save_model")
    print("-------------------------")
    
    
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
#     cwd = os.getcwd()
#     print("----------------------------")
#     print("current dir: ", cwd)

#     s3_client = boto3.client("s3")
#     prefix = "hackathon"
        
#     print("-------------------------")
#     bucket = "sagemaker-hackathon-demo-eu-west-1-017233837209"
    
#     #client.put_object(Body=more_binary_data, Bucket='my_bucket_name', Key='my/key/including/anotherfilename.txt')
    
#     inputs = s3_client.put_object(, bucket, "model.pth")
    
#     s3 = session.resource('s3')

#     result = s3.Bucket(bucket).upload_file(os.path.join(model_dir, "model.pth"),'hackathon/model.pth')
#     print("inputs_model_saved", inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    print("-------------------------")
    print("__main")
    print("-------------------------")

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer for training.")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        metavar="DROP",
        help="dropout rate (default: 0.5)",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        metavar="KERNEL",
        help="conv2d filter kernel size (default: 5)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=10,
        help="number of channels in hidden conv layer",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args = parser.parse_args()

    train(args)
