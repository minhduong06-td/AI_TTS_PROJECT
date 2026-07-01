import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def _reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "none":
        return x
    if reduction == "mean":
        return x.mean(dim=0)
    if reduction == "sum":
        return x.sum(dim=0)
    raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _validate_input(
    tensors: list[torch.Tensor],
    dim_range: tuple[int, int] = (0, -1),
    data_range: tuple[float, float] = (0.0, -1.0),
    size_range: tuple[int, int] | None = None,
) -> None:
    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert isinstance(t, torch.Tensor), f"Expected torch.Tensor, got {type(t)}"
        assert t.device == x.device, f"Expected tensors to be on {x.device}, got {t.device}"

        if size_range is None:
            assert t.size() == x.size(), f"Expected tensors with same size, got {t.size()} and {x.size()}"
        else:
            assert t.size()[size_range[0] : size_range[1]] == x.size()[size_range[0] : size_range[1]], (
                f"Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}"
            )

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f"Expected number of dimensions to be {dim_range[0]}, got {t.dim()}"
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], (
                f"Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}"
            )

        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), f"Expected values to be greater or equal to {data_range[0]}, got {t.min()}"
            assert t.max() <= data_range[1], f"Expected values to be lower or equal to {data_range[1]}, got {t.max()}"


def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0

    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 11,
    kernel_sigma: float = 1.5,
    data_range: int | float = 1.0,
    reduction: str = "mean",
    full: bool = False,
    downsample: bool = True,
    k1: float = 0.01,
    k2: float = 0.03,
) -> list[torch.Tensor]:
    assert kernel_size % 2 == 1, f"Kernel size must be odd, got [{kernel_size}]"
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    x = x / float(data_range)
    y = y / float(data_range)
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    if full:
        return [ssim_val, cs]

    return ssim_val


class SSIMLoss(_Loss):
    __constants__ = ["kernel_size", "k1", "k2", "sigma", "kernel", "reduction"]

    def __init__(
        self,
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        downsample: bool = True,
        reduction: str = "mean",
        data_range: int | float = 1.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, f"Kernel size must be odd, got [{kernel_size}]"
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        score = ssim(
            x=x,
            y=y,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            downsample=self.downsample,
            data_range=self.data_range,
            reduction=self.reduction,
            full=False,
            k1=self.k1,
            k2=self.k2,
        )
        return torch.ones_like(score) - score


def _ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. Input size: {x.size()}. Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x**2
    mu_yy = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def _ssim_per_channel_complex(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. Input size: {x.size()}. Kernel size: {kernel.size()}"
        )
    c1 = k1**2
    c2 = k2**2
    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map
    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs
