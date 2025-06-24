import torch
import torch.optim as optim
from model import VGG
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import save_image
from utils import load_image


def main():

    # model = models.vgg19(pretrained=True).features
    # print(model)
    # selected layers, from paper are:
    #   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 356
    transform_ = Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
        ]
    )

    # prepare images
    original_image = load_image("sample-image.jpg", device, image_size, transform_)
    style_image = load_image("style.jpg", device, image_size, transform_)
    generated = original_image.clone().requires_grad_(True)

    # hyperparams
    total_steps = 6000
    learning_rate = 0.001
    alpha = 1
    beta = 0.01
    optimizer = optim.Adam([generated], lr=learning_rate)

    model = VGG().to(device).eval()

    for step in range(total_steps):
        generated_features = model(generated)
        original_image_features = model(original_image)
        style_features = model(style_image)

        style_loss = content_loss = 0

        for gen_feature, orig_feature, style_feature in zip(
            generated_features, original_image_features, style_features
        ):
            batch_size, channel, height, width = gen_feature.shape
            content_loss += torch.mean((gen_feature - orig_feature) ** 2)

            # compute Gram matrix,
            # which is (once more) a kinda correlation matrix
            # with pixel values on both images having similar values meaning,
            # that style is similar
            G = gen_feature.view(channel, height * width).mm(
                gen_feature.view(channel, height * width).t()
            )
            A = style_feature.view(channel, height * width).mm(
                style_feature.view(channel, height * width).t()
            )
            # .. or Hyena-operator

            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print("Total loss: ", total_loss.item())
            save_image(generated, f"generated_{step:03d}.png")


if __name__ == "__main__":
    main()
