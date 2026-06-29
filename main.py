from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from train import train_classifier, test_classifier, train_vae, test_vae
from model import LatentClassifier, LatentConVAE, ConVAE, Classifier
device = "cuda:0"

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: x.view(-1)
        )
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    model = LatentClassifier(z_dim=256)
    model = model.to(device)
    train_classifier(train_loader, model, 100, "latent_classifier")
    test_classifier(train_loader, model)
    test_classifier(test_loader, model)
    

# if __name__ == "__main__":
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(
#             lambda x: x.view(-1)
#         )
#     ])

#     train_dataset = datasets.MNIST(
#         root="./data",
#         train=True,
#         download=True,
#         transform=transform
#     )

#     test_dataset = datasets.MNIST(
#         root="./data",
#         train=False,
#         download=True,
#         transform=transform
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=64,
#         shuffle=True
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=64,
#         shuffle=False
#     )
#     model = Classifier()
#     model = model.to(device)
#     train_classifier(train_loader, model, 100, "classifier.pth")
#     test_classifier(test_loader, model)

# if __name__ == "__main__":
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     train_dataset = datasets.FashionMNIST(
#         root="./data",
#         train=True,
#         download=True,
#         transform=transform,
#     )
#     test_dataset = datasets.FashionMNIST(
#         root="./data",
#         train=False,
#         download=True,
#         transform=transform,
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=64,
#         shuffle=True,
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=64,
#         shuffle=False,
#     )
#     model = LatentConVAE(z_dim=1280)
#     model = model.to(device)
#     train_vae(train_loader, model, 100, "./latent_con_vae.pth", beta=1e-2)
#     test_vae(test_loader, model)

# if __name__ == "__main__":
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#     train_dataset = datasets.FashionMNIST(
#         root="./data",
#         train=True,
#         download=True,
#         transform=transform,
#     )
#     test_dataset = datasets.FashionMNIST(
#         root="./data",
#         train=False,
#         download=True,
#         transform=transform,
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=64,
#         shuffle=True,
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=64,
#         shuffle=False,
#     )
#     model = ConVAE()
#     model = model.to(device)
#     train_vae(train_loader, model, 100, "./vae.pth")
#     test_vae(test_loader, model)