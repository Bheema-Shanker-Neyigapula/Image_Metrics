import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import torch
from torchvision import transforms, models
import torch.nn.functional as F

class ImageMetrics:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError("Image file could not be loaded.")
        self.image_tensor = self._prepare_image_for_model()

    def _prepare_image_for_model(self):
        """Prepare the image for deep learning model input."""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def calculate_mse(self):
        """Calculate Mean Squared Error (MSE) for the image (self-comparison as an example)."""
        return np.mean((self.image - self.image) ** 2)

    def calculate_psnr(self):
        """Calculate Peak Signal-to-Noise Ratio (PSNR) for the image (self-comparison as an example)."""
        mse = self.calculate_mse()
        if mse == 0:
            return float('inf')  # PSNR is infinite when images are identical
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    def calculate_ssim(self):
        """Calculate Structural Similarity Index (SSIM) for the image (self-comparison as an example)."""
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return ssim(image_gray, image_gray)

    def calculate_entropy(self):
        """Calculate the entropy of the image's grayscale histogram."""
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(image_gray.ravel(), bins=256, range=(0, 256), density=True)
        return entropy(hist)

    def calculate_variance(self):
        """Calculate the variance of the image."""
        return np.var(self.image)

    def calculate_mean_intensity(self):
        """Calculate the mean intensity of the image."""
        return np.mean(self.image)

    def calculate_histogram_uniformity(self):
        """Calculate histogram uniformity as a measure of distribution evenness."""
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(image_gray.ravel(), bins=256, range=(0, 256))
        hist_normalized = hist / hist.sum()
        return np.sum(hist_normalized ** 2)

    def calculate_feature_similarity(self):
        """Calculate feature similarity using a pre-trained deep learning model."""
        model = models.resnet18(pretrained=True)
        model.eval()
        with torch.no_grad():
            features = model(self.image_tensor.unsqueeze(0))
        return features.norm().item()

    def calculate_inception_score(self):
        """Calculate the Inception Score."""
        model = models.inception_v3(pretrained=True)
        model.eval()
        with torch.no_grad():
            logits = model(self.image_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            marginal_probs = np.mean(probs, axis=0)
            kl_div = probs * (np.log(probs + 1e-10) - np.log(marginal_probs + 1e-10))
            score = np.exp(np.sum(kl_div) / probs.shape[0])
        return score

    def calculate_spectral_norm(self):
        """Calculate spectral norm of the image for analyzing high-frequency components."""
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        fft_image = np.fft.fft2(image_gray)
        return np.max(np.abs(fft_image))

    def display_metrics(self):
        """Print all computed metrics for the image."""
        print(f"Mean Squared Error (MSE): {self.calculate_mse():.2f}")
        print(f"Peak Signal-to-Noise Ratio (PSNR): {self.calculate_psnr():.2f} dB")
        print(f"Structural Similarity Index (SSIM): {self.calculate_ssim():.4f}")
        print(f"Entropy: {self.calculate_entropy():.4f}")
        print(f"Variance: {self.calculate_variance():.4f}")
        print(f"Mean Intensity: {self.calculate_mean_intensity():.4f}")
        print(f"Histogram Uniformity: {self.calculate_histogram_uniformity():.4f}")
        print(f"Feature Similarity (ResNet-18): {self.calculate_feature_similarity():.4f}")
        print(f"Spectral Norm: {self.calculate_spectral_norm():.4f}")
        print(f"Inception Score: {self.calculate_inception_score():.4f}")

if __name__ == "__main__":
    # Path to the example image
    image_path = "apple.jpg"

    # Initialize the ImageMetrics class and display metrics
    metrics = ImageMetrics(image_path)
    metrics.display_metrics()
