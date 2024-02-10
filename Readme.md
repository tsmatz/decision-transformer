# Decision Transformer Example (Atari Pong)

This example shows you Decision Transformer implementation (and step-by-step explanation in notebook) with introductory Atari Pong game environment.<br>

Unlike [official example](https://github.com/kzl/decision-transformer), this example is runnable in the mainstream computes with small footprint - such as, a signle GPU of Tesla T4 or consumer GPUs (NVIDIA RTX) - for you to try this code easily.

## Prerequisites

Before running, please set up Python and GPU drivers (CUDA).<br>
In my case, I have used Ubuntu Server 20.04 LTS in Microsoft Azure and run the following command.


```
# install tools for compiling
sudo apt-get update
sudo apt install -y gcc
sudo apt-get install -y make

# download and install cuda
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
sudo sh cuda_11.7.1_515.65.01_linux.run

# pip setup
sudo apt-get update
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip
```

This example uses a part of large dataset, [here](https://research.google/resources/datasets/dqn-replay/), for training.<br>
To download this dataset, we need to install ```gsutil```. (Refer [official document](https://cloud.google.com/storage/docs/gsutil_install) for details about installation of ```gsutil```.)

```
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Install required packages to run this example.

```
pip3 install torch numpy matplotlib opencv-python atari_py
```

In this example, we use an Atari environment in ```atari_py``` package.<br>
Download and import ROMs for running this environment.

```
sudo apt-get install unrar
wget http://www.atarimania.com/roms/Roms.rar
unrar x -r Roms.rar
python3 -m atari_py.import_roms ROMS
```

Install jupyter to run notebook.

```
pip3 install jupyter
```

> Note : In this example, we need to download large dataset from Google, but we only use a part of this dataset.<br>
> Please remove dataset which is not used in this example, or [expand your disk](https://learn.microsoft.com/en-us/azure/virtual-machines/linux/expand-disks) not to exceed disk spaces.

## Run

Start jupyter notebook (see below) and run [dt_atari_pong.ipynb](./dt_atari_pong.ipynb).

```
jupyter notebook
```
