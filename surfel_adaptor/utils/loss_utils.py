import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from lpips import LPIPS
import pynanoflann

def smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def huber_log_loss(scales: torch.Tensor,
                   delta: float = 0.5,
                   eps: float = 1e-12,
                   reduction: str = "mean") -> torch.Tensor:
    sx = scales[:, 0].clamp_min(eps)
    sy = scales[:, 1].clamp_min(eps)
    r = torch.log(sx / sy)

    abs_r = r.abs()
    loss = torch.where(abs_r <= delta,
                       0.5 * (abs_r ** 2) / delta,
                       abs_r - 0.5 * delta)

    return loss.mean() if reduction == "mean" else loss.sum()

def knn_search(queried_pc : torch.Tensor, K, max_radius=1e10):
    queried_pc_np = queried_pc.cpu().detach().numpy()

    # 初始化KDTree，设置参数
    nn = pynanoflann.KDTree(
        n_neighbors = K + 1,  # 查询K+1个邻居（包含自身）
        metric = 'L2',  # 使用L2距离（欧氏距离）
        radius = max_radius  # 设置足够大的搜索半径确保覆盖所有点
    )
    nn.fit(queried_pc_np)  # 拟合点云数据

    # 执行KNN查询（查询集与数据集相同）
    distances, indices = nn.kneighbors(queried_pc_np)

    # 排除第一个索引（自身），转换为PyTorch张量，并移至原设备
    knn_indices = torch.from_numpy(indices[:, 1:]).long().to(queried_pc.device)
    knn_distances = torch.from_numpy(distances[:, 1:]).float().to(queried_pc.device)

    # condition = knn_indices > queried_pc.shape[0]
    # indices = torch.where(condition)
    # idx_values = knn_indices[condition]
    # dis_values = knn_distances[condition]
    # for i, j, id_val, dis_val in zip(indices[0], indices[1], idx_values, dis_values):
    #     print(f"位置 ({i}, {j}) - 索引值: {id_val.item():.4f} - 距离值: {dis_val.item():.4f} - 对应的坐标: {queried_pc[indices[0]]} ")

    return knn_indices, knn_distances

# knn: knn points xyz, knn_n: knn points normal (N x k x 3)
def xyz_loss(src, knn_pc, knn_n, w):
    src_usq = torch.unsqueeze(src, 1) # src_usq: (N x 1 x 3) knn: (N x k x 3)
    p_mu = src_usq-knn_pc
    dis = torch.einsum('nki,nki->nk', p_mu, knn_n).abs() # (N x k)
    return (w * dis).mean()

def normal_loss(src_n, knn_n, w):
    k = knn_n.shape[1]
    w = torch.unsqueeze(w, -1)
    n_bar = torch.nn.functional.normalize(torch.sum(w * knn_n, dim=1), dim=1) # (N x 1 x 3)
    # n_bar.squeeze(1)
    nTn = torch.einsum('nk,nk->nk', src_n, n_bar)  # (N x 1)
    return (torch.ones_like(nTn) - nTn).abs().mean()
    #! Equally contribution of neibor surfuel
    # nTn = torch.einsum('nki,nki->nk', src_n.unsqueeze(1), knn_n) # (N x k)
    # mean = torch.sum(nTn, dim=1)/k
    # return (torch.ones_like(mean)-mean).abs().mean()

def scale_loss(src_s, knn_s, src_vs, knn_vs, knn_norm, w, max_scale=1):
    # Renturn the weighted distance from point-to-plane
    # src: (N x 3), knn, knn_n: (N, k, 3), weight: (N, k, 3)
    def plane_dis(src, knn, knn_n, weight):
        src_usq = torch.unsqueeze(src, 1)  # src_usq: (N x 1 x 3)
        p_mu = src_usq - knn
        dis = torch.einsum('nki,nki->nk', p_mu, knn_n).abs()  # (N x k)
        return (w * dis).mean()
    src_su = src_s[:, 0]
    src_sv = src_s[:, 1]
    over_u = torch.any(src_su > max_scale)
    over_v = torch.any(src_sv > max_scale)
    if over_u | over_v:
      src_vsu = src_vs[:, 0, :]
      src_vsv = src_vs[:, 1, :]
      knn_vsu = knn_vs[:, :, 0, :]
      knn_vsv = knn_vs[:, :, 1, :]
      if over_u & over_v:
        return plane_dis(src_vsu, knn_vsu, knn_norm, w) + plane_dis(src_vsv, knn_vsv, knn_norm, w)
      elif over_u:
        return plane_dis(src_vsu, knn_vsu, knn_norm, w)
      elif over_v:
        return plane_dis(src_vsv, knn_vsv, knn_norm, w)
    else:
        return 0.

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()


def normal_angle(pred, gt):
    pred = pred * 2.0 - 1.0
    gt = gt * 2.0 - 1.0
    norms = pred.norm(dim=-1) * gt.norm(dim=-1)
    cos_sim = (pred * gt).sum(-1) / (norms + 1e-9)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos_sim[norms > 1e-9])).mean()
    if ang.isnan():
        return -1
    return ang
