import torch, torchvision
import numpy as np
from src.dataloader import ImageDataset
from tqdm import tqdm
from scipy import linalg



class FIDEval:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inception_v3 = torchvision.models.inception_v3(pretrained = True)
        inception_v3.fc = torch.nn.Identity()
        self.inception_v3 = inception_v3.to(self.device)

    def infer_inception(self, images):
        with torch.no_grad():
            f = self.inception_v3(images.to(self.device))[0]
        return f


    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


    def infer_features(self, image_loader, desc = ""):
        train_pbar = tqdm(image_loader, desc = desc)

        features = []
        for images in train_pbar:
            f = self.infer_inception(images.to(self.device))
            features.append(f.detach().cpu())
        
        return torch.concat(features, dim = 0).numpy()


    def fid_score(self, fake_image_loader, real_image_loader):
        real_features = self.infer_features(real_image_loader, 
                                            desc = "Inferring features for real images")
        fake_features = self.infer_features(fake_image_loader,
                                            desc = "Inferring features for fake images")

        mean_real = np.mean(real_features, axis = 0)
        mean_fake = np.mean(fake_features, axis = 0)

        cov_real = np.cov(real_features, rowvar = False)
        cov_fake = np.cov(fake_features, rowvar = False)

        score = self.calculate_frechet_distance(mean_real, cov_real, mean_fake, cov_fake)

        return score