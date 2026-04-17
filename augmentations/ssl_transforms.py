import torchvision.transforms as T

class SimCLRAugment:
    """Dual-view augmentation for SimCLR"""
    def __init__(self, size=224):
        self.aug = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.aug(x), self.aug(x)

# add the MAE masking logic here later!