import numpy as np
import cv2

def gaussian2D(sigma: tuple[float, float], kernel_size: tuple[int, int]) -> np.ndarray:
    """
    Args:
        sigma: a tuple of x, y scales (standard deviations)
        kernel_size: a tuple of x, y dimensions of the kernel

    Returns:
        returns a 2D gaussian blur kernel
    """
    # code here
    gauss = np.ndarray([])
    g1 = []
    g2 = []
    for x in range(kernel_size[0]):
        g1.append(1 / (np.sqrt(2 * np.pi) * sigma[0]) * np.exp(- ((x - kernel_size[0] // 2) ** 2) / (2 * sigma[0] ** 2)))
    for y in range(kernel_size[1]):
        g2.append(1 / (np.sqrt(2 * np.pi) * sigma[1]) * np.exp(- ((y - kernel_size[1] // 2) ** 2) / (2 * sigma[1] ** 2)))

    gauss = np.outer(g1, g2)
    return gauss

def ssim(image1, image2, multichannel=True):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    Args:
        image1: First image.
        image2: Second image.
        multichannel: If True, treat the last dimension as channels.
    Returns:
        SSIM score between the two images.
    """

    assert image1.shape == image2.shape, "Images must have the same shape"
    assert image1.shape[-1] == image2.shape[-1], "Images must have the same number of channels"

    img1_float = image1.astype(np.float32)
    img2_float = image2.astype(np.float32)

    '''
    SSIM(x, y) = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)               
                 / ((mu_x^2 + mu_y^2 + c1) * (sigma_x^2 + sigma_y^2 + c2))

    mu_x, mu_y: pixel sample means
    sigma_x, sigma_y: sample variance
    sigma_xy: sample covariance

    c1, c2: two variables to stabilize the division with weak denominator
    L: dynamic range of the pixel values (255 for 8-bit grayscale images)
    k1, k2 = 0.01, 0.03 by default
    '''
    k1, k2 = 0.01, 0.03
    L = 255.0

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Gaussian window
    sigma = 1.5
    kernel_size = 11
    gauss2d = gaussian2D((sigma, sigma), (11, 11))
    gauss2d = gauss2d / np.sum(gauss2d)

    # Calculate means (mu)
    mu1 = cv2.filter2D(img1_float, -1, gauss2d)
    mu2 = cv2.filter2D(img2_float, -1, gauss2d)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Calculate variance and covariance
    sigma1_sq = cv2.filter2D(img1_float ** 2, -1, gauss2d) - mu1_sq
    sigma2_sq = cv2.filter2D(img2_float ** 2, -1, gauss2d) - mu2_sq
    sigma12 = cv2.filter2D(img1_float * img2_float, -1, gauss2d) - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 0) # Ensure non-negative variance due to floating point errors
    sigma2_sq = np.maximum(sigma2_sq, 0) # Ensure non-negative variance due to floating point errors

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    mean_ssim = np.mean(ssim_map) # Mean SSIM over the entire image
    return mean_ssim

def rmse(image1, image2):
    '''
    Calculate the Root Mean Squared Error (RMSE) between two images.
    Args:
        image1: First image.
        image2: Second image.
    Returns:
        RMSE score between the two images.
    '''

    assert image1.shape == image2.shape, "Images must have the same shape"

    img1_float = image1.astype(np.float32)
    img2_float = image2.astype(np.float32)
    rmse = np.sqrt(np.mean((img1_float - img2_float) ** 2))
    return rmse

def cosine_similarity(image1, image2):
    '''
    Calculate the cosine similarity between two images.
    Args:
        image1: First image.
        image2: Second image.
    Returns:
        Cosine similarity score between the two images.
    '''

    assert image1.shape == image2.shape, "Images must have the same shape"

    img1_float = image1.flatten()
    img2_float = image2.flatten()

    dot_prod = np.dot(img1_float, img2_float)
    norm1 = np.linalg.norm(img1_float)
    norm2 = np.linalg.norm(img2_float)
    norm_product = norm1 * norm2
    if norm_product < 1e-9:
        return 0.0  # Avoid division by zero
    else:
        similarity = dot_prod / norm_product
        return np.clip(similarity, -1.0, 1.0)
    
def cosine_distance(image1, image2):
    '''
    Calculate the cosine distance between two images.
    Args:
        image1: First image.
        image2: Second image.
    Returns:
        Cosine distance score between the two images.
    '''
    return 1 - cosine_similarity(image1, image2)
