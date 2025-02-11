from PIL import Image
import torchvision


def load_image(
    image_name: str,
    device: str,
    image_size: int,
    loader: torchvision.transforms.Compose,
):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)
