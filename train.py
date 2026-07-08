import warnings
import torch
from torch.utils import data
from pathlib import Path

# Custom module imports
from config import TRAIN_IMG_DIR, VAL_IMG_DIR, NUM_CLASSES, INPUT_SIZE, BATCH_SIZE, EPOCHS, LR, MOMENTUM, PARAMS
from dataset import SimpleDataset
from model import yolo_v26_n
from loss import ComputeLoss
from val import validate

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    
    # 1. Dataset prep
    dataset = SimpleDataset(TRAIN_IMG_DIR, INPUT_SIZE, PARAMS, augment=True)
    loader  = data.DataLoader(dataset, BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True,
                               collate_fn=SimpleDataset.collate_fn)

    # 2. Model initialization
    model = yolo_v26_n(NUM_CLASSES).cuda()

    # 3. Loss function setup
    loss_fn = ComputeLoss(model, PARAMS, EPOCHS)

    # 4. Optimizer setup
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                                 weight_decay=PARAMS["weight_decay"], nesterov=True)

    # 5. Training execution
    Path("weights").mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for samples, targets in loader:
            samples = samples.cuda().float() / 255   

            preds = model(samples)                   
            loss  = loss_fn(preds, targets)          

            optimizer.zero_grad()
            loss.backward()                          
            optimizer.step()                         

            total_loss += loss.item()

        loss_fn.step()                               
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}]  avg loss: {avg_loss:.4f}", end="")

        if (epoch + 1) % 10 == 0:
            map50 = validate(model, VAL_IMG_DIR, INPUT_SIZE, BATCH_SIZE)
            print(f"  |  mAP@50: {map50:.4f}", end="")

        print()

    torch.save({"model": model.state_dict()}, "weights/last.pt")
    print("Done. Saved to weights/last.pt")