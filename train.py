import torch
from train_utils import *
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader = prepare_dataloaders("./Task01_BrainTumour", batch=1)

models = {
    "swinunetr": create_model("swinunetr", device)
}

for name, model in models.items():

    print(f"\n🔥 Training {name}")

    opt = torch.optim.Adam(model.parameters(), 1e-4)
    loss_fn = DiceLoss(sigmoid=True)
    metric = DiceMetric(include_background=False)

    best_dice = 0
    patience = 10
    counter = 0

    train_losses = []
    val_dices = []

    for epoch in range(300):

        loss = train_one_epoch(train_loader, model, opt, loss_fn, device)
        dice = validate(val_loader, model, metric, device)

        train_losses.append(loss)
        val_dices.append(dice)

        print(f"{name} | Epoch {epoch} | Loss={loss:.4f} | Dice={dice:.4f}")

        # ✅ save best model
        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), f"{name}_best.pth")
            counter = 0
        else:
            counter += 1

        # ✅ early stopping
        if counter >= patience:
            print("⛔ Early stopping")
            break

    # 📊 plot
    plt.figure()
    plt.plot(train_losses, label="Loss")
    plt.plot(val_dices, label="Dice")
    plt.legend()
    plt.title(name)
    plt.savefig(f"{name}_curve.png")