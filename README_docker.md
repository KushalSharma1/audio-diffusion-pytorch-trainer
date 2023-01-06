# Instructions for Docker

- docker build -t ma_dyna_tt .
- docker run -it --gpus all --net host --ipc host ma_dyna_tt bash
- conda create -n ma_dyna_TT_venv python=3.8-y
- conda activate ma_dyna_TT_venv
- pip install -r requirements.txt
- git clone https://github.com/KushalSharma1/audio-diffusion-pytorch-trainer.git
- cd audio-diffusion-pytorch-trainer/
- git checkout 0.0.43