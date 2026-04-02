import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
from utils import util_common
import lpips
from torchvision import transforms

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    ## borrow from InvSR （CVPR2025）
    # latent LPIPS loss
    # if 'llpips' not in cfg:
    #     raise ValueError("Missing required key 'llpips' in configuration.")
    # params = cfg.llpips.get('params', dict)
    # llpips_loss = util_common.get_obj_from_str(cfg.llpips.target)(**params)
    # llpips_loss.cuda()

    # for params in llpips_loss.parameters():
    #     params.requires_grad = False
    #
    # # loading the pre-trained model
    # ckpt_path = cfg.llpips.ckpt_path
    # lpips_weights = torch.load(ckpt_path, map_location="cpu") #["state_dict"]
    # llpips_loss.load_state_dict(lpips_weights, strict=True)

    lpips_vgg = lpips.LPIPS(net='vgg').cuda()
    lpips_vgg.eval()

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    for name, param in cldm.named_parameters():
        param.requires_grad = False


    for name, param in cldm.named_parameters():
        if 'Adjust_blocks' in name:
            param.requires_grad = True
            print(" Activate parameters ", name)
        else:
            param.requires_grad = False


    total_model_params = sum(p.numel() for p in cldm.vae.parameters())
    trainable_model_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, cldm.vae.parameters()))
    print(f"Model parameters: Total={total_model_params}, Trainable={trainable_model_params}")
    trainable_params = filter(lambda p: p.requires_grad, cldm.vae.parameters())
    # Setup optimizer:
    opt = torch.optim.AdamW(trainable_params, lr=cfg.train.learning_rate)

    if cfg.train.resume:
        checkpoint = torch.load(cfg.train.resume, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            cldm.load_controlnet_from_ckpt(checkpoint['model_state_dict'])
        else:
            cldm.load_controlnet_from_ckpt(checkpoint)

        # if 'optimizer_state_dict' in checkpoint:
        #     opt.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'global_step' in checkpoint:
            global_step = checkpoint['global_step']
        else:
            global_step = 0
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    if cfg.train.resume_vae:
        checkpoint = torch.load(cfg.train.resume_vae, map_location="cpu")
        if 'model_state_dict' in checkpoint:
            cldm.load_vae_from_ckpt(checkpoint['model_state_dict'])
        else:
            cldm.load_vae_from_ckpt(checkpoint)
        print(
            f"load VAE weight from checkpoint: {cfg.train.resume_vae}"
        )

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)


    # Prepare models for training:
    cldm.train().to(device)
    # swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    # Variables for monitoring/logging purposes:
    max_steps = cfg.train.train_steps
    step_Total_loss = []
    step_mse_base_loss = []
    step_mse_loss = []
    step_lpips_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    mse_loss = nn.MSELoss()

    target_size = (256, 256)
    resize_transform = transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR)

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, z_pred = batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            B, _, _, _ = gt.shape

            # print("gt shape", gt.shape)
            # print("z_pred shape", z_pred.shape)

            with torch.no_grad():
                z_l, enc_feats = pure_cldm.vae_encode(lq, Stage2=True)
                # cond = pure_cldm.prepare_condition(lq)
                # cond_aug = copy.deepcopy(cond)
                # random_step = torch.randint(5, 50, (1,)).item()
                # z_pred = sampler.sample(
                #     model=cldm,
                #     device=device,
                #     steps=50,
                #     x_size=(len(lq), *z_l.shape[1:]),
                #     cond=cond_aug,
                #     uncond=None,
                #     cfg_scale=1.0,
                #     progress=False, # accelerator.is_main_process,
                # )
            # 1. 随机初始化 FP_weights, 用于平衡 MSE 和 LPIPS
            # FP_weights = torch.rand(1).to(device)    ## # 用 Sigmoid 平滑 FP_weights   Fedility and Perception weights
            # FP_weights = torch.randint(0, 11, (B,), device=device, dtype=torch.float32) / 10.0
            # FP_weights = torch.randint(0, 11, (B,), device=device, dtype=torch.float32)
            # FP_weights = 0.9 * torch.rand(1, device="cuda") + 0.1  ## 产生 0.1 到1

            # FP_weights_mapped = exponential_mapping(FP_weights, k=1.0) ## 转换到 0.01 ~ 1.0

            out_en = pure_cldm.vae_decode(z_pred, enc_feats=enc_feats, weight=1)
            loss_mse = mse_loss(out_en, gt)

            # out_en_resized = resize_transform(out_en)
            # gt_resized = resize_transform(gt)
            # loss_lpips = lpips_vgg(torch.clamp(out_en_resized, min=-1.0, max=1.0), gt_resized).mean() * cfg.lpips.coef
            loss_lpips = torch.zeros_like(loss_mse)

            # lambda_MSE = FP_weights_mapped
            # lambda_LPIPS = 1.0 - FP_weights_mapped

            # beta = 0.02
            loss_mse_val = loss_mse
            loss_lpips_val = loss_lpips
            loss_total = loss_mse_val + loss_lpips_val  # 保底 MSE + 动态平衡 MSE 和 LPIPS

            # if global_step % 100 == 0:  # 每100步打印一次
            #     print(f"[Step {global_step}] FP_weight: {FP_weights_mapped.item():.2f}, "
            #           f"MSE: {(loss_mse).item():.6f}, LPIPS: {(loss_lpips).item():.6f}, "
            #           f"LPIPS_weight: {loss_lpips_val.item():.6f}, ")

            opt.zero_grad()
            accelerator.backward(loss_total)
            torch.nn.utils.clip_grad_norm_(cldm.parameters(), max_norm=1.0)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_Total_loss.append(loss_total.item())
            # step_mse_base_loss.append(loss_mse_base.item())
            step_mse_loss.append(loss_mse_val.item())
            step_lpips_loss.append(loss_lpips_val.item())

            epoch_loss.append(loss_total.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, L_total: {loss_total.item():.6f},   L_mse: {loss_mse_val.item():.6f}, L_per: {loss_lpips_val.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_Total_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                # avg_mse_base_loss = (
                #     accelerator.gather(
                #         torch.tensor(step_mse_base_loss, device=device).unsqueeze(0)
                #     )
                #     .mean()
                #     .item()
                # )
                avg_mse_loss = (
                    accelerator.gather(
                        torch.tensor(step_mse_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                avg_lpips_loss = (
                    accelerator.gather(
                        torch.tensor(step_lpips_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                step_Total_loss.clear()
                # step_mse_base_loss.clear()
                step_mse_loss.clear()
                step_lpips_loss.clear()

                if accelerator.is_main_process:
                    writer.add_scalar("loss/Total_loss_step", avg_loss, global_step)
                    writer.add_scalar("loss/MSE_loss_step", avg_mse_loss, global_step)
                    writer.add_scalar("loss/LPIPS_loss_step", avg_lpips_loss, global_step)
                    # writer.add_scalar("loss/Base_loss_step", avg_mse_base_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        'model_state_dict': pure_cldm.vae.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'global_step': global_step,
                        'epoch': epoch
                    }
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}_VAE_EN.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 2
                # log_clean = clean[:N]
                # log_cond = {k: v[:N] for k, v in cond.items()}
                # log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_z_pred = z_pred[:N]
                log_enc = [feat[:N] for feat in enc_feats]
                # log_prompt = prompt[:N]
                # print("log_gt shape", log_gt.shape, "enc_feats[:N]", enc_feats[:N])

                # cldm.eval()
                # with torch.no_grad():
                #     z_pred = sampler.sample(
                #         model=cldm,
                #         device=device,
                #         steps=50,
                #         x_size=(len(log_gt), *z_l.shape[1:]),
                #         cond=log_cond,
                #         uncond=None,
                #         cfg_scale=1.0,
                #         progress=accelerator.is_main_process,
                #     )

                if accelerator.is_main_process:
                    for tag, image in [
                        ("image/enhance", (pure_cldm.vae_decode(log_z_pred, enc_feats=log_enc, weight=0.5) + 1) / 2),
                        ("image/gt", (log_gt + 1) / 2),
                        ("image/lq", log_lq),
                        ("image/samples", (pure_cldm.vae_decode(log_z_pred) + 1) / 2),
                        # (
                        #     "image/condition_decoded",
                        #     (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                        # ),
                        # (
                        #     "image/condition_aug_decoded",
                        #     (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                        # ),
                        # (
                        #     "image/prompt",
                        #     (log_txt_as_img((64, 64), log_prompt) + 1) / 2,
                        # ),
                    ]:
                        writer.add_image(tag, make_grid(image, nrow=4), global_step)
                # cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()




def exponential_mapping(
        x: torch.Tensor,
        x_min: float = 0.1,
        x_max: float = 1.0,
        y_min: float = 0.001,
        y_max: float = 1.0,
        k: float = 5.0,
        power: int = 3
) -> torch.Tensor:
    """
    改进版指数映射函数 (支持GPU/梯度/类型安全)

    特性：
    1. 自动对齐输入张量的设备与数据类型
    2. 严格类型检查与参数验证
    3. 支持反向传播计算
    4. 显式内存管理

    参数：
    x       : 输入张量 (任意形状)
    x_min   : 输入值下限
    x_max   : 输入值上限 (需 > x_min)
    y_min   : 输出值下限
    y_max   : 输出值上限 (需 > y_min)
    k       : 指数曲率因子 (建议 >0)
    power   : 末端加速因子 (整数 >=1)

    返回：
    torch.Tensor : 映射后的张量，保持输入形状
    """
    # 参数校验
    assert x_max > x_min, "x_max must be greater than x_min"
    assert y_max > y_min, "y_max must be greater than y_min"
    assert isinstance(power, int) and power >= 1, "power must be integer >=1"

    # 自动获取输入张量的设备与数据类型
    device = x.device
    dtype = x.dtype

    # 将标量参数转换为与输入匹配的Tensor
    def to_tensor(val):
        return torch.tensor(val, dtype=dtype, device=device)

    x_min_t = to_tensor(x_min)
    x_max_t = to_tensor(x_max)
    y_min_t = to_tensor(y_min)
    y_max_t = to_tensor(y_max)
    k_t = to_tensor(k)

    # 安全截断与归一化
    x_clipped = torch.clamp(x, min=x_min_t, max=x_max_t)
    normalized = (x_clipped - x_min_t) / (x_max_t - x_min_t + 1e-8)  # 防止除零

    # 幂次变换 (末端加速)
    powered = 1 - (1 - normalized).pow(power)

    # 指数映射核心计算
    numerator = torch.exp(k_t * powered) - 1
    denominator = torch.exp(k_t) - 1  # 使用Tensor计算

    # 线性缩放输出
    scaled_output = (numerator / (denominator + 1e-8)) * (y_max_t - y_min_t) + y_min_t

    # 内存优化 (分离不需要梯度的部分)
    return torch.where(
        (x >= x_min_t) & (x <= x_max_t),
        scaled_output,
        to_tensor(y_min if x < x_min else y_max)
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
