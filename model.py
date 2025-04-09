import torch
from torch import nn
import torch.nn.functional as F
import math
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import numpy as np


class SeperableConv2d(nn.Module):
    """
    Một lớp tích chập khả tách, gồm tích chập theo chiều sâu và tích chập theo điểm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super(SeperableConv2d, self).__init__()
        
        # Tích chập theo chiều sâu
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels, 
            bias=bias,
            padding=padding
        )
        
        # Tích chập theo điểm
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels, 
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    
    
class ConvBlock(nn.Module):
    """
    Khối tích chập, bao gồm lớp tích chập khả tách, chuẩn hóa theo batch và hàm kích hoạt.
    """
    def __init__(self, in_channels, out_channels, use_act=True, use_bn=True, discriminator=False, **kwargs):
        super(ConvBlock, self).__init__()
        
        # Có sử dụng hàm kích hoạt hay không
        self.use_act = use_act

        # Lớp tích chập khả tách
        self.cnn = SeperableConv2d(in_channels, out_channels, **kwargs, bias=not use_bn)

        # Lớp chuẩn hóa theo batch
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Hàm kích hoạt
        self.act = nn.LeakyReLU(0.2, inplace=True) if discriminator else nn.PReLU(num_parameters=out_channels)
        
    def forward(self, x):
        if self.use_act:
            res = self.act(self.bn(self.cnn(x)))
        else:
            res = self.bn(self.cnn(x))
        return res


class UpsampleBlock(nn.Module):
    """
    Khối tăng mẫu, sử dụng Pixel Shuffle để tăng độ phân giải.
    """
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        
        # Lớp tích chập khả tách
        self.conv = SeperableConv2d(in_channels, in_channels * scale_factor**2, kernel_size=3, stride=1, padding=1)
        
        # Lớp Pixel Shuffle để tái sắp xếp pixel
        self.ps = nn.PixelShuffle(scale_factor)
        
        # Hàm kích hoạt PReLU
        self.act = nn.PReLU(num_parameters=in_channels)
    
    def forward(self, x):
        return self.act(self.ps(self.conv(x)))
        

class ResidualBlock(nn.Module):
    """
    Khối dư, gồm hai khối tích chập và một kết nối dư.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        # Khối tích chập đầu tiên với hàm kích hoạt
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Khối tích chập thứ hai không có hàm kích hoạt
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False
        )
        
    def forward(self, x):
        # Đầu ra của khối tích chập đầu tiên
        out = self.block1(x)
        
        # Đầu ra của khối tích chập thứ hai
        out = self.block2(out)
        
        # Cộng với đầu vào ban đầu để tạo kết nối dư
        return out + x


class Generator(nn.Module):
    """
    Mô hình Generator cho việc siêu phân giải ảnh.
    """
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16, upscale_factor=4):
        super(Generator, self).__init__()
        
        # Khối tích chập đầu tiên
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        
        # Chuỗi các khối dư
        self.residual = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        # Khối tích chập trung gian
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        
        # Các khối tăng mẫu
        self.upsampler = nn.Sequential(
            *[UpsampleBlock(num_channels, scale_factor=2) for _ in range(upscale_factor//2)]
        )

        # Lớp tích chập cuối cùng
        self.final_conv = SeperableConv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        
    def forward(self, x):
        # Lưu lại đầu ra của khối đầu tiên để thực hiện kết nối dư toàn cục
        initial = self.initial(x)
        
        # Đi qua chuỗi các khối dư
        x = self.residual(initial)
        
        # Đi qua khối tích chập trung gian và cộng với đầu ra của khối đầu tiên
        x = self.convblock(x) + initial
        
        # Đi qua các khối tăng mẫu
        x = self.upsampler(x)

        # Đi qua lớp tích chập cuối cùng và chuẩn hóa đầu ra về khoảng [0, 1]
        return (torch.tanh(self.final_conv(x)) + 1) / 2


def gaussian(window_size, sigma):
    """
    Tạo một cửa sổ Gaussian 1D.
    
    Args:
        window_size (int): Kích thước cửa sổ
        sigma (float): Độ lệch chuẩn
        
    Returns:
        torch.Tensor: Cửa sổ Gaussian đã được chuẩn hóa
    """
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Tạo một cửa sổ Gaussian 2D.
    
    Args:
        window_size (int): Kích thước cửa sổ
        channel (int): Số kênh
        
    Returns:
        torch.Tensor: Cửa sổ Gaussian 2D
    """
    # Tạo cửa sổ Gaussian 1D
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    
    # Tạo cửa sổ Gaussian 2D bằng cách nhân ngoài hai cửa sổ 1D
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    # Mở rộng cửa sổ 2D cho tất cả các kênh
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Tính chỉ số SSIM giữa hai ảnh.
    
    Args:
        img1 (torch.Tensor): Ảnh thứ nhất
        img2 (torch.Tensor): Ảnh thứ hai
        window (torch.Tensor): Cửa sổ Gaussian
        window_size (int): Kích thước cửa sổ
        channel (int): Số kênh
        size_average (bool): Có lấy trung bình hay không
        
    Returns:
        torch.Tensor: Chỉ số SSIM
    """
    # Tính giá trị trung bình của hai ảnh
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # Tính bình phương giá trị trung bình
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    
    # Tính tích giá trị trung bình
    mu1_mu2 = mu1 * mu2

    # Tính phương sai
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Các hằng số
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Tính SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # Lấy trung bình nếu cần
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Hàm tính chỉ số SSIM giữa hai ảnh.
    
    Args:
        img1 (torch.Tensor): Ảnh thứ nhất
        img2 (torch.Tensor): Ảnh thứ hai
        window_size (int): Kích thước cửa sổ
        size_average (bool): Có lấy trung bình hay không
        
    Returns:
        torch.Tensor: Chỉ số SSIM
    """
    # Lấy kích thước ảnh
    (_, channel, _, _) = img1.size()

    # Tạo cửa sổ Gaussian
    window = create_window(window_size, channel)
    
    # Kiểm tra xem có sử dụng GPU hay không
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    
    # Chuyển cửa sổ về cùng kiểu dữ liệu với ảnh
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(img1, img2):
    """
    Tính chỉ số PSNR giữa hai ảnh.
    
    Args:
        img1 (torch.Tensor): Ảnh thứ nhất
        img2 (torch.Tensor): Ảnh thứ hai
        
    Returns:
        float: Chỉ số PSNR
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse)


def load_model(model_path, device=None):
    """
    Tải mô hình từ file .pth.
    
    Args:
        model_path (str): Đường dẫn đến file mô hình
        device (torch.device): Thiết bị để chạy mô hình
        
    Returns:
        nn.Module: Mô hình đã được tải
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo mô hình
    model = Generator().to(device)
    
    # Tải trọng số
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Chuyển sang chế độ đánh giá
    model.eval()
    
    return model


def process_image(model, image, device=None):
    """
    Xử lý ảnh với mô hình Generator.
    
    Args:
        model (nn.Module): Mô hình Generator
        image (PIL.Image): Ảnh đầu vào
        device (torch.device): Thiết bị để chạy mô hình
        
    Returns:
        tuple: (Ảnh đầu ra, Số điểm ảnh đầu vào, Số điểm ảnh đầu ra)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lưu lại độ phân giải ảnh đầu vào
    input_resolution = image.width * image.height
    
    # Tiền xử lý: chuyển ảnh sang tensor và chuẩn hóa
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Tắt gradient để tăng tốc độ và tiết kiệm bộ nhớ
    with torch.no_grad():
        output = model(img_tensor)
    
    # Hậu xử lý: chuyển tensor về ảnh
    output_img = transforms.ToPILImage()(output.squeeze(0).cpu())
    
    # Lưu lại độ phân giải ảnh đầu ra
    output_resolution = output_img.width * output_img.height
    
    return output_img, input_resolution, output_resolution


def calculate_metrics(original_tensor, enhanced_tensor, device=None):
    """
    Tính toán các chỉ số chất lượng ảnh.
    
    Args:
        original_tensor (torch.Tensor): Tensor ảnh gốc
        enhanced_tensor (torch.Tensor): Tensor ảnh đã được làm nét
        device (torch.device): Thiết bị để tính toán
        
    Returns:
        dict: Các chỉ số chất lượng ảnh
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chuyển tensor về cùng thiết bị
    original_tensor = original_tensor.to(device)
    enhanced_tensor = enhanced_tensor.to(device)
    
    # Tính PSNR
    psnr_value = psnr(original_tensor, enhanced_tensor).item()
    
    # Tính SSIM
    ssim_value = ssim(original_tensor.unsqueeze(0), enhanced_tensor.unsqueeze(0)).item()
    
    return {
        'psnr': psnr_value,
        'ssim': ssim_value
    }