import torch
import torch.nn.functional as F
import torch.nn as nn
device = "cuda:0"

def train_classifier(train_loader, model, epoch, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(epoch):
        model.train()
        train_loss = 0
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            loss =F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f"Epoch{e} loss: {train_loss}")
    torch.save( model.state_dict(), save_path )

def test_classifier(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
    accuracy = correct/total
    return accuracy

def train_vae(train_loader, model, epoch, save_path, beta=1e-7):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(epoch):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        for x, _ in train_loader:
            x = x.to(device)
            outputs = model(x)
            recon = outputs["recon"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            recon_loss = F.binary_cross_entropy(recon,x,reduction="mean",)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        total_loss /= len(train_loader)
        total_recon /= len(train_loader)
        total_kl /= len(train_loader)
        print(
            f"Epoch {e} | "
            f"loss: {total_loss:.6f} | "
            f"recon: {total_recon:.6f} | "
            f"kl: {total_kl:.6f}"
        )
    torch.save(model.state_dict(),save_path)

def test_vae(test_loader, model, beta=1e-4):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            outputs = model(x)
            recon = outputs["recon"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    total_loss /= len(test_loader)
    total_recon /= len(test_loader)
    total_kl /= len(test_loader)
    print(
        f"Test | "
        f"loss: {total_loss:.6f} | "
        f"recon: {total_recon:.6f} | "
        f"kl: {total_kl:.6f}"
    )