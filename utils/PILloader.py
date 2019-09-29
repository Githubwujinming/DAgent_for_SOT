
import torchvision.transforms as transforms


loader = transforms.Compose([
transforms.ToTensor()])

unloader = transforms.ToPILImage()