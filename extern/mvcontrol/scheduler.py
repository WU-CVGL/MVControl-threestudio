import torch
from diffusers import DDIMScheduler, DDPMScheduler

class DDIMScheduler_(DDIMScheduler):
    """
    Add a method to reconstruct clean image based on the predicted noise
    which is an inverse of `add_noise`
    Notice that it's origial DDPM formulation
    """
    def pred_x0(
        self,
        sample: torch.FloatTensor,
        timesteps: torch.IntTensor,
        model_output: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # 1. device
        device = timesteps.device

        # 2. compute alphas, betas
        alphas_cumprod = self.alphas_cumprod.to(device)
        alpha_prod_t = alphas_cumprod[timesteps]

        beta_prod_t = 1 - alpha_prod_t

        if timesteps.ndim > 0:
            beta_prod_t = beta_prod_t[:, None, None, None]
            alpha_prod_t = alpha_prod_t[:, None, None, None]

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # # 4. Clip or threshold "predicted x_0"
        # if self.config.thresholding:
        #     pred_original_sample = self._threshold_sample(pred_original_sample)
        # elif self.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.config.clip_sample_range, self.config.clip_sample_range
        #     )
        return pred_original_sample
    

        
