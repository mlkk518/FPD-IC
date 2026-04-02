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
from collections import OrderedDict

from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.sampler import SpacedSampler
from utils import util_common
from GAN.vqgan_arch import VQGANDiscriminator
from GAN.losses.losses import GANLoss

def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
    recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
    return d_weight


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



    ## borrow from InvSR （CVPR2025）
    # latent LPIPS loss
    if 'llpips' not in cfg:
        raise ValueError("Missing required key 'llpips' in configuration.")
    params = cfg.llpips.get('params', dict)
    llpips_loss = util_common.get_obj_from_str(cfg.llpips.target)(**params)
    llpips_loss.cuda()

    for params in llpips_loss.parameters():
        params.requires_grad = False

    # loading the pre-trained model
    ckpt_path = cfg.llpips.ckpt_path
    lpips_weights = torch.load(ckpt_path, map_location="cpu") #["state_dict"]
    llpips_loss.load_state_dict(lpips_weights, strict=True)

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
    if cfg.Gan_train.get('Gan_enable'):
        Gan_loss = GANLoss(gan_type="hinge").to(device)
        net_d = VQGANDiscriminator(nc=4, ndf=128, n_layers=4).to(device)
        if cfg.Gan_train.ckpt_path is not None:
            ckpt_path = cfg.Gan_train.ckpt_path
            if ckpt_path != "":
                print("Loading Discriminator model from ", ckpt_path)
                checkpoint = torch.load(ckpt_path, map_location=f"cuda:0")
                print("checkpoint  keys ===", checkpoint.keys())
                # Removing 'module.' prefix to match model's state_dict structure
                new_state_dict = {k.replace('module.', ''): v for k, v in
                                  checkpoint["state_dict"].items()}  # state_dict  params

                try:
                    # Load state dict with strict=False to ignore missing keys
                    net_d.load_state_dict(new_state_dict, strict=False)
                    print("Successfully loaded the checkpoint")
                except Exception as e:
                    print(f"Error loading the state_dict: {e}")
        net_d.train()
        optimizer_d = torch.optim.AdamW([
            {'params': net_d.parameters(), 'lr': cfg.Gan_train.lr},
        ])

        # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)


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

        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        else:
            epoch = 0

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

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)

    total_model_params = sum(p.numel() for p in cldm.controlnet.parameters())
    trainable_model_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, cldm.controlnet.parameters()))
    print(f"Model parameters: Total={total_model_params}, Trainable={trainable_model_params}")



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
    if cfg.Gan_train.get('Gan_enable'):
        net_d, optimizer_d = accelerator.prepare(net_d, optimizer_d)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    # noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    max_steps = cfg.train.train_steps
    step_Total_loss = []
    step_LDM_loss = []
    step_Lpips_loss = []


    step_g_GAN_loss = []



    epoch_loss = []
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

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
            gt, lq = batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

            # optimize net_g
            if cfg.Gan_train.get('Gan_enable'):
                for p in net_d.parameters():
                    p.requires_grad = False

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                cond = pure_cldm.prepare_condition(lq)
                cond_aug = copy.deepcopy(cond)
                # if noise_aug_timestep > 0:
                #     cond_aug["c_img"] = diffusion.q_sample(
                #         x_start=cond_aug["c_img"],
                #         t=torch.randint(
                #             0, noise_aug_timestep, (z_0.shape[0],), device=device
                #         ),
                #         noise=torch.randn_like(cond_aug["c_img"]),
                #     )
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss_ldm, model_output, target = diffusion.p_losses(cldm, z_0, t, cond_aug)
            # loss_llpips = llpips_loss(model_output, target).mean() * cfg.llpips.coef
            # loss_llpips = torch.zeros_like(loss_ldm)


            if global_step > cfg.Gan_train.get('net_g_start_iter') and cfg.Gan_train.get('Gan_enable'):
                fake_g_pred = net_d(model_output)
                l_g_gan = Gan_loss(fake_g_pred, True, is_disc=False)
                d_weight = cfg.Gan_train.get('scale_adaptive_gan_weight')
                l_g_gan *= d_weight
            else:
                l_g_gan = torch.zeros_like(loss_ldm)

            loss_total = loss_ldm #+ loss_llpips + l_g_gan

            opt.zero_grad()
            accelerator.backward(loss_total)
            opt.step()

            accelerator.wait_for_everyone()

            # optimize net_d
            loss_dict = OrderedDict()
            if global_step > cfg.Gan_train.get('net_g_start_iter') and cfg.Gan_train.get('Gan_enable'):
                for p in net_d.parameters():
                    p.requires_grad = True
                optimizer_d.zero_grad()

                real_d_pred = net_d(target)
                l_d_real = Gan_loss(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                accelerator.backward(l_d_real)
                # fake
                fake_d_pred = net_d(model_output.detach())
                l_d_fake = Gan_loss(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                accelerator.backward(l_d_fake)
                optimizer_d.step()
                accelerator.wait_for_everyone()


            global_step += 1
            step_Total_loss.append(loss_total.item())
            step_LDM_loss.append(loss_ldm.item())
            # step_Lpips_loss.append(loss_llpips.item())

            if global_step > cfg.Gan_train.get('net_g_start_iter') and cfg.Gan_train.get('Gan_enable'):
                step_g_GAN_loss.append(l_g_gan.item())
                # step_l_d_real_loss.append(loss_dict['l_d_real'].item())
                # step_out_d_real_loss.append(loss_dict['out_d_real'].item())
                # step_l_d_fake_loss.append(loss_dict['l_d_fake'].item())
                # step_out_d_fake_loss.append(loss_dict['out_d_fake'].item())

            epoch_loss.append(loss_total.item())
            pbar.update(1)

            if cfg.Gan_train.get('Gan_enable'):
                # 设置进度条后缀
                pbar.set_postfix({
                    "Epoch": f"{epoch:04d}",
                    "Global Step": f"{global_step:07d}",
                    "L_total": f"{loss_total.item():.6f}",
                    "L_ldm": f"{loss_ldm.item():.6f}",
                    # "L_lpips": f"{loss_llpips.item():.6f}",
                    "L_gan": f"{l_g_gan.item():.6f}",
                    "L_d_real": f"{loss_dict['l_d_real'].item():.6f}",
                    # "Out_d_real": f"{loss_dict['out_d_real'].item():.6f}",
                    # "L_d_fake": f"{loss_dict['l_d_fake'].item():.6f}",
                    # "Out_d_fake": f"{loss_dict['out_d_fake'].item():.6f}"
                })
            else:
                pbar.set_description(
                    # f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, L_total: {loss_total.item():.6f}, L_ldm: {loss_ldm.item():.6f}"
                    f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, L_total: {loss_total.item():.6f}, L_ldm: {loss_ldm.item():.6f}" #L_lpips: {loss_llpips.item():.6f}
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

                avg_LDM_loss = (
                    accelerator.gather(
                        torch.tensor(step_LDM_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                # avg_Lpips_loss = (
                #     accelerator.gather(
                #         torch.tensor(step_Lpips_loss, device=device).unsqueeze(0)
                #     )
                #     .mean()
                #     .item()
                # )

                if cfg.Gan_train.get('Gan_enable'):
                    avg_Gan_loss = (
                        accelerator.gather(
                            torch.tensor(step_g_GAN_loss, device=device).unsqueeze(0)
                        )
                        .mean()
                        .item()
                    )
                    step_g_GAN_loss.clear()


                step_Total_loss.clear()
                step_LDM_loss.clear()
                # step_Lpips_loss.clear()


                if accelerator.is_main_process:
                    writer.add_scalar("loss/Total_loss_step", avg_loss, global_step)
                    writer.add_scalar("loss/LDM_loss_step", avg_LDM_loss, global_step)
                    # writer.add_scalar("loss/Llpips_loss_step", avg_Lpips_loss, global_step)
                    if cfg.Gan_train.get('Gan_enable'):
                        writer.add_scalar("loss/avg_Gan_loss_step", avg_Gan_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    print(" Save the trained model!")
                    if cfg.Gan_train.get('Gan_enable'):
                        checkpoint = {
                            'model_state_dict': pure_cldm.controlnet.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch
                        }
                        ckpt_path = f"{ckpt_dir}/net_g_{global_step:07d}.pt"
                        torch.save(checkpoint, ckpt_path)

                        checkpoint_netd = {
                            'model_state_dict': net_d.state_dict(),
                            'optimizer_state_dict': optimizer_d.state_dict(),
                        }
                        ckpt_path = f"{ckpt_dir}/net_d_{global_step:07d}.pt"
                        torch.save(checkpoint_netd, ckpt_path)
                    else:
                        checkpoint = {
                            'model_state_dict': pure_cldm.controlnet.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch
                        }
                        ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                        torch.save(checkpoint, ckpt_path)
                    # checkpoint = pure_cldm.controlnet.state_dict()
                    # ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    # torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 4
                # log_clean = clean[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                # log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                # log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            # ("image/condition", log_clean),
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
                            #     (log_txt_as_img((32, 32), log_prompt) + 1) / 2,
                            # ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
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
            writer.add_scalar("loss/loss_Epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
