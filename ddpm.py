import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
from tqdm import tqdm
import tool
import h5py

LAMB=0.5

def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    timesteps: int, start: int = 3, end: int = 3, tau: int = 1
) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2 - 1


def unnormalize_to_zero_to_one(img: torch.Tensor) -> torch.Tensor:
    return (img + 1) * 0.5


def identity(x: torch.Tensor) -> torch.Tensor:
    return x
def extract(
    constants: torch.Tensor, timestamps: torch.Tensor, shape: int
) -> torch.Tensor:
    batch_size = timestamps.shape[0]
    constants=constants.to(timestamps.device)
    out = constants.gather(-1, timestamps).to(timestamps.device)
    # print(out.device)
    return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(timestamps.device)


class DiffusionModel(nn.Module):
    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(
        self,
        model: nn.Module,
        image_size: list,
        pos_file_path: str,
        pos_index: tuple,
        model_path: str = None,
        classes: int = 10,
        pos: bool = False,
        num_classes: int = 2,
        cond: bool = False,
        beta_scheduler: str = "linear",
        timesteps: int = 1000,
        schedule_fn_kwargs: dict | None = None,
        auto_normalize: bool = True,
        sample_num: int = 5,
        device=None,
        batch_size=1,
    ) -> None:
        super().__init__()
        self.model = model
        self.pos = pos
        self.classes = classes
        self.pos_index = pos_index
        self.pos_file_path = pos_file_path
        self.channels = 1
        self.image_size = image_size
        self.model_path = model_path
        self.sample_num = sample_num
        self.NUM_CLASSES = num_classes
        self.device = device
        self.cond = cond
        self.batch_size = batch_size

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        self.betas = self.beta_scheduler_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", self.betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        timesteps, *_ = self.betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = timesteps

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def target_index(
        self,
    ):
        (i, j, k) = tool.choice_target_using_pos(
            D_min=150, D_max=160, file_path=self.pos_file_path
        )
        return (i, j, k)

    @torch.inference_mode()
    def p_sample(
        self,
        x: torch.Tensor,
        timestamp: int,
        cond_concat: bool = False,
        cond_list: list = None,
    ) -> torch.Tensor:
        b, *_, device = *x.shape, x.device
        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )
        # print(x.shape)
        # if x.shape[1]==1:
        if cond_list is not None:
            preds = self.model(x, batched_timestamps, cond_list)
        else:
            preds = self.model(x, batched_timestamps)
            # print("pred shape",preds.shape)
        betas_t = extract(self.betas, batched_timestamps, x.shape)
        sqrt_recip_alphas_t = extract(
            self.sqrt_recip_alphas, batched_timestamps, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
        )
        if self.pos:
            (x, _) = torch.split(x, 1, dim=1)
        elif cond_concat:
            assert (
                x.shape[1] == 2
            ), f"the shape of x must be 2,but the shape of your x is {x.shape[1]}"
            (x, _) = torch.split(x, 1, dim=1)
        else:
            x = x
        # preds = self.model(x, batched_timestamps)
        predicted_mean = sqrt_recip_alphas_t * (
            x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
        )
        if timestamp == 0:
            return predicted_mean
        else:
            posterior_variance = extract(
                self.posterior_variance, batched_timestamps, x.shape
            )
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.inference_mode()
    def p_sample_loop(
        self,
        shape: tuple,
        return_all_timesteps: bool = False,
        train: bool = False,
        x_cond: torch.Tensor = None,
        cond: bool = False,
        cond_list: list = None,
        concat:bool=False,
        interpolate:bool=False,
    ) -> torch.Tensor:
        # batch, device = shape[0], "cuda"

        img = torch.randn(shape, device=self.device, requires_grad=True)
        if interpolate:
            tgt_image=torch.randn(shape, device=self.device, requires_grad=True)
            img=LAMB*img+(1-LAMB)*tgt_image
        # print(img.shape)
        imgs = [img]
        if train:
            for t in reversed(range(0, self.num_timesteps)):
                assert (
                    img.shape[1] == 1
                ), f"the shape of img must be 1,but the shape of your img is {img.shape[1]}"
                if self.pos:
                    pos_file = h5py.File(self.pos_file_path, "r")["pos"][:]
                    (i, j, k) = self.pos_index
                    pos_data = tool.process_pos(
                        pos_file[i, j, k, :], device=self.device
                    )
                    pos_data = pos_data.expand(
                        (self.sample_num, *pos_data.shape)
                    ).unsqueeze(1)
                    assert (
                        pos_data.shape == img.shape
                    ), f"their shape should be the same,but the pos shape is{pos_data.shape},img shape is{img.shape}"
                    img = torch.cat((img, pos_data), dim=1)
                    if x_cond is not None and concat:
                        img = torch.cat((img, x_cond), dim=1)
                        # print(1)
                    # print(img.shape)
                else:
                    if x_cond is not None and concat:
                        # print(img.shape)
                        img = torch.cat((img, x_cond), dim=1)
                        assert (
                            img.shape[1] == 2
                        ), f"the shape of imf must be 2,but the shape of your img is {img.shape[1]}"
                    img = img
                if cond:
                    assert cond_list is not None, f"the cond_list is None"
                    img = self.p_sample(img, t, cond_list)
                    img = img.clip(-1, 1)
                else:
                    if x_cond is not None and concat:
                        img = self.p_sample(
                            img, t, cond_list=cond_list, cond_concat=True
                        )
                    else:
                        img = self.p_sample(img, t, cond_list=cond_list)
                    # print('sample shape:',img.shape)
                torch.cuda.empty_cache()
                imgs.append(img)
                # print('done')
        else:
            for t in tqdm(
                reversed(range(0, self.num_timesteps)), total=self.num_timesteps
            ):
                assert (
                    img.shape[1] == 1
                ), f"the shape of img must be 1,but the shape of your img is {img.shape[1]}"
                if self.pos:
                    pos_file = h5py.File(self.pos_file_path, "r")["pos"][:]
                    (i, j, k) = self.pos_index
                    pos_data = tool.process_pos(
                        pos_file[i, j, k, :], device=self.device
                    )
                    pos_data = pos_data.expand(
                        (self.sample_num, *pos_data.shape)
                    ).unsqueeze(1)
                    assert (
                        pos_data.shape == img.shape
                    ), f"their shape should be the same,but the pos shape is{pos_data.shape},img shape is{img.shape}"
                    img = torch.cat((img, pos_data), dim=1)
                    if x_cond is not None:
                        img = torch.cat((img, x_cond), dim=1)
                        # print(1)
                    # print(img.shape)
                else:
                    if x_cond is not None:
                        # print(img.shape)
                        img = torch.cat((img, x_cond), dim=1)
                        assert (
                            img.shape[1] == 2
                        ), f"the shape of imf must be 2,but the shape of your img is {img.shape[1]}"
                    img = img
                if cond:
                    assert cond_list is not None, f"the cond_list is None"
                    img = self.p_sample(img, t, cond_list)
                    img = img.clip(-1, 1)
                else:
                    if x_cond is not None:
                        img = self.p_sample(
                            img, t, cond_list=cond_list, cond_concat=True
                        )
                    else:
                        img = self.p_sample(img, t, cond_list=cond_list)
                    # print('sample shape:',img.shape)
                torch.cuda.empty_cache()
                imgs.append(img)
                # print('done')
        ret = imgs  # if not return_all_timesteps else torch.stack(imgs, dim=1)

        # ret = self.unnormalize(ret)
        return ret

    def sample(
        self,
        batch_size,
        return_all_timesteps: bool = False,
        train: bool = False,
        cond: bool = False,
        x_cond: torch.Tensor = None,
        cond_list: list = None,
        concat:bool=False,
        interpolate:bool=False,
    ) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size[0], self.image_size[1])
        return self.p_sample_loop(
            shape,
            return_all_timesteps=return_all_timesteps,
            train=train,
            x_cond=x_cond,
            cond=cond,
            cond_list=cond_list,
            concat=concat,
            interpolate=interpolate,
        )

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def pred_x0_from_noise(self, pred, x_t, t):
        if self.pos:
            (x_t, _) = torch.split(x_t, 1, dim=1)
            assert pred.shape == x_t.shape
        else:
            assert pred.shape == x_t.shape
            x_t = x_t
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        x_0 = (x_t - sqrt_one_minus_alphas_cumprod_t * pred) / sqrt_alphas_cumprod_t
        return x_0

    def p_loss(
        self,
        x_start: torch.Tensor,
        t: int,
        noise: torch.Tensor = None,
        x_cond: torch.Tensor = None,
        concat: bool = False,
        interpolate: bool = False,
        loss_type: str = "l2",
    ) -> torch.Tensor:

        if self.pos:
            assert x_start.shape[1] == 2
            (data, pos) = torch.split(x_start, 1, dim=1)
            if noise is None:
                noise = torch.randn_like(data, device=data.device)
            else:
                noise = noise.to(data.device)
            # print(data.shape)
            x_noised = self.q_sample(data, t, noise=noise)
            # print(x_noised.shape)
            assert (
                x_noised.shape == pos.shape
            ), f"x_noise shape is {x_noised.shape},but pos shape is {pos.shape}"
            x_noised = torch.cat((x_noised, pos), dim=1)
            if x_cond is not None and concat:
                assert (
                    x_noised.device == x_cond.device
                ), f"the device of x_noised and x_cond must be the same,but the device of x_noised is {x_noised.device},the device of x_cond is {x_cond.device}"
                x_noised = torch.cat((x_noised, x_cond), dim=1)
            elif x_cond is not None and interpolate:
                assert (
                    x_noised.device == x_cond.device
                ), f"the device of x_noised and x_cond must be the same,but the device of x_noised is {x_noised.device},the device of x_cond is {x_cond.device}"
                cond_noised = self.q_sample(x_cond, t, noise=noise)
                def slerp(z1, z2, alpha):
                    theta = torch.acos(
                        torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2))
                    )
                    return (
                        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                        + torch.sin(alpha * theta) / torch.sin(theta) * z2
                    )
                x_noised = LAMB * x_noised + (1 - LAMB) * cond_noised
            else:
                pass
            predicted_noise = self.model(x_noised, t)
        else:
            if noise is None:
                noise = torch.randn_like(x_start, device=x_start.device)
            else:
                noise = noise.to(x_start.device)
            x_noised = self.q_sample(x_start, t, noise=noise)
            if x_cond is not None and concat:
                assert (
                    x_noised.device == x_cond.device
                ), f"the device of x_noised and x_cond must be the same,but the device of x_noised is {x_noised.device},the device of x_cond is {x_cond.device}"
                x_noised = torch.cat((x_noised, x_cond), dim=1)
            elif x_cond is not None and interpolate:
                assert (
                    x_noised.device == x_cond.device
                ), f"the device of x_noised and x_cond must be the same,but the device of x_noised is {x_noised.device},the device of x_cond is {x_cond.device}"
                cond_noised = self.q_sample(x_cond, t, noise=noise)

                def slerp(z1, z2, alpha):
                    theta = torch.acos(
                        torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2))
                    )
                    return (
                        torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                        + torch.sin(alpha * theta) / torch.sin(theta) * z2
                    )

                x_noised = LAMB * x_noised + (1 - LAMB) * cond_noised
            else:
                pass
            predicted_noise = self.model(x_noised, t)
        if loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"unknown loss type {loss_type}")
        return loss

    def p_loss_cond(
        self,
        x_start: torch.Tensor,
        t: int,
        cond_list: list,
        noise: torch.Tensor = None,
        loss_type: str = "l2",
    ) -> torch.Tensor:
        if self.pos:
            assert x_start.shape[1] == 2
            (data, pos) = torch.split(x_start, 1, dim=1)
            if noise is None:
                noise = torch.randn_like(data, device=data.device)
            else:
                noise = noise.to(data.device)
            # print(data.shape)
            x_noised = self.q_sample(data, t, noise=noise)
            # print(x_noised.shape)
            assert (
                x_noised.shape == pos.shape
            ), f"x_noise shape is {x_noised.shape},but pos shape is {pos.shape}"
            x_noised = torch.cat((x_noised, pos), dim=1)
            predicted_noise = self.model(x_noised, t, cond_list)
        else:
            if noise is None:
                noise = torch.randn_like(x_start, device=x_start.device)
            else:
                noise = noise.to(x_start.device)
            x_noised = self.q_sample(x_start, t, noise=noise)
            predicted_noise = self.model(x_noised, t, cond_list)
        if loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"unknown loss type {loss_type}")
        return loss

    def gen_pos_idx(
        self,
    ):
        _, idx = tool.pos_index_random_1(self.pos_file_path)
        return idx

    def forward(
        self,
        x: torch.Tensor,
        cond: bool,
        cond_list: list = None,
        x_cond: torch.Tensor = None,
        concat:bool=False,
        interpolate:bool=False,
    ) -> torch.Tensor:
        # b,c,h,w, device, [img_size] = *x.shape, x.device, self.image_size
        # print(x.shape)
        [b, c, h, w] = x.shape
        # print(h,w)
        assert h == self.image_size[0], f"image size must be {self.image_size[0]}"
        assert w == self.image_size[1], f"image size must be {self.image_size[1]}"
        timestamp = (
            torch.randint(0, self.num_timesteps, (self.batch_size,))
            .long()
            .to(self.device)
        )
        # print(timestamp.device)
        # x = self.normalize(x)
        if cond:
            return self.p_loss_cond(x, timestamp, cond_list)
        else:
            return self.p_loss(x, timestamp, x_cond=x_cond,concat=concat,interpolate=interpolate)


if __name__ == "__main__":
    import model
    model_=model.Unet(image_channels=1)
    model_.to('cuda')
    cond=[torch.randn(2,32,6336,device='cuda'),torch.randn(2,64,100,device='cuda'),torch.randn(2,256,100,device='cuda')]
    #model_unet=model.Unet(image_channels=1)
    #model_unet=model_unet.to('cuda')
    ddpm=DiffusionModel(model=model_,
                            image_size=[1050,90],
                            device='cuda',
                            pos_index=(0,0,0),
                            pos=False,
                            pos_file_path=None,
                            batch_size=2,)
    x=torch.randn((2,1,1050,90)).to('cuda')
    x_cond=torch.randn((2,1,1050,90)).to('cuda')
    y=ddpm.sample(batch_size=2,train=True,cond=False,cond_list=None,x_cond=None,concat=False,interpolate=True)
    print(y.shape)
