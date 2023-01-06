import torch 
from main.module_base import Model
#from audio_diffusion_pytorch import AudioDiffusionModel, UniformDistribution, LinearSchedule, KarrasSchedule, VSampler
# from audio_diffusion_pytorch import LinearSchedule, VSampler
from audio_diffusion_pytorch import AudioDiffusionModel, KarrasSchedule, AEulerSampler, LogNormalDistribution
import torchaudio
import math 

adm = AudioDiffusionModel(
    in_channels=2,
    channels=256,
    patch_factor=32,
    patch_blocks=1,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attentions=[0, 0, 0, 1, 1, 1],
    attention_heads=16,
    attention_features=64,
    attention_multiplier=2,
    use_nearest_upsample=False,
    use_skip_scale=True,
    # use_magnitude_channels=True,
    diffusion_sigma_distribution=LogNormalDistribution(-3, 1)
)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device = 'cpu'
print(f"Using device {device}")
adm = adm.to(device)
model_path = "logs/2022-12-23-02-20-50/epoch=3599-valid_loss=0.018.ckpt" #"logs/ckpts/2022-12-20-00-45-45/epoch=37-valid_loss=0.053.ckpt"
model = Model.load_from_checkpoint(
    checkpoint_path=model_path,
    learning_rate=1e-4,
    beta1=0.95,
    beta2=0.999,
    in_channels=2,
    channels=256,
    patch_factor=32,
    patch_blocks=1,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    multipliers=[1, 2, 4, 4, 4, 4, 4],
    factors=[4, 4, 4, 2, 2, 2],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attentions=[0, 0, 0, 1, 1, 1],
    attention_heads=16,
    attention_features=64,
    attention_multiplier=2,
    use_nearest_upsample=False,
    use_skip_scale=True,
    # use_magnitude_channels=True,
    diffusion_sigma_distribution=LogNormalDistribution(-3, 1)
).to(device)

sampling_rate = 48000
# @markdown Generation length in seconds (will be rounded to be a power of 2 of sample_rate*length)
length = 10 #@param {type: "slider", min: 1, max: 87, step: 1}
length_samples = math.ceil(math.log2(length * sampling_rate))
# @markdown Number of samples to generate 
num_samples = 10 #@param {type: "slider", min: 1, max: 16, step: 1}
# @markdown Number of diffusion steps (higher tends to be better but takes longer to generate)
num_steps = 500 #@param {type: "slider", min: 1, max: 200, step: 1}

#import pdb; pdb.set_trace()
with torch.no_grad():
    samples = model.model.sample(
        noise=torch.randn((num_samples, 2, 2 ** length_samples), device=device),
        num_steps=num_steps,
        sigma_schedule=KarrasSchedule(
            sigma_min=1e-4, 
            sigma_max=10.0,
            rho=7.0
        ),
        sampler=AEulerSampler(),
    )

# Log audio samples 
for i, sample in enumerate(samples):
    cpu_sample = sample.cpu()
    torchaudio.save(f'./audio_sample_{i}.wav', cpu_sample, sampling_rate)