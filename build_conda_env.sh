conda create -n vggt python=3.10 -y
conda activate vggt
cd vggt_multi

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements_demo.txt
python -m pip install -e .

# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # H200

python -m pip install hydra-core omegaconf iopath wcmatch yacs fvcore tensorboard rich wandb
python -m pip install smplx
conda install -c conda-forge chumpy
