import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import torch.nn as nn


def view_dataloader_images(dataloader, n=8):
    if n > 8:
        print(f"Having n higher than 10 will create messy plots, lowering to 10.")
        n = 8
    imgs, labels = next(iter(dataloader))

    grid_img = torchvision.utils.make_grid(imgs, nrow=n)
    plt.imshow(grid_img.numpy().permute(1, 2, 0))
    plt.show()


def pred_and_plot_image(model,
                        image_path: str, 
                        class_names,
                        image_size,
                        transform=None,
                        device='cpu'):
    
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


    model.to(device)

    model.eval()

    with torch.inference_mode():
      
      transformed_image = image_transform(img).unsqueeze(dim=0).to(device)
      target_image_pred = model(transformed_image)

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)


