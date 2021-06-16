
import torch
ROOT = "C:\\Users\\lotan\\Documents\\studies\\Affordances\\datasets\\test"
import torchvision
import matplotlib.pyplot as plt
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess_func = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
preprocess = lambda input_data: preprocess_func(input_data).unsqueeze(0).to(device)


t = torchvision.datasets.ImageFolder(ROOT, preprocess)
d = torch.utils.data.DataLoader(dataset=t, batch_size=len(t), shuffle=False, num_workers=4)

model_o = torchvision.models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model_o.children())[:-1])

print("i")


# import clip
# from PIL import Image
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)