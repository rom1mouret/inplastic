from hashlib import sha1
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def path_to_number(path: str) -> int:
    directory = path.rsplit("/", 1)[0]
    return int(sha1(directory.encode("utf-8")).hexdigest(), 16)


def train_selector(path: str) -> bool:
    return path_to_number(path) % 10 < 4  # 4 out of 10 directories


def test_selector1(path: str) -> bool:
    n = path_to_number(path) % 10
    return n >= 4 and n <= 6  # 3 out of 10 directories


def test_selector2(path: str) -> bool:
    return not train_selector(path) and not test_selector1(path)


def load_images(selector, max_per_class: int) -> list:
    train_dataset = ImageFolder(
                        root="fruits-360/Training",
                        transform=transforms.ToTensor(),
                        is_valid_file=selector)
    dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True
    )

    print("loading dataset in memory...")
    class_to_img = defaultdict(list)
    max_n = 0
    with tqdm(total=max_per_class+1) as progress_bar:
        for img, label in dataloader:
            li = class_to_img[label.item()]
            if len(li) < max_per_class:
                li.append(img)
                progress_bar.update(max(0, len(li) - max_n))
                max_n = max(max_n, len(li))
        progress_bar.update(1)

    return list(class_to_img.values())
