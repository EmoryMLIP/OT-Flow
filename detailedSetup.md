# Detailed Set-up and Running Instructions

## Set-up

go to local folder where you want the files and type 
```
git init
```

copy all files from the remote repository into this local folder:
```
git pull git@github.com:EmoryMLIP/OT-Flow
```

Set vim as the default editor (this step often just helps on linux):
```
git config --global core.editor "vim"
```

Create a virtual environment (may need to install virtualenv command) to hold all the python package versions for this project:
```
virtualenv -p python3 otEnv
```

Start up the virtual environment:
```
source otEnv/bin/activate
```

If you're running python 3.7 (and pip 21.3.1), install all the requirements:
```
pip install -r requirements.txt 
```

We used Python 3.5 and CUDA 9.2, so we installed pytorch separately via
```
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

For full capabilities, set the values in config.py to match your architecture.


### Training and Evaluating Toy Data Sets
commands with hyperparameters


```
python trainToyOTflow.py --data 8gaussians --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,30.0,1 --niters 5000 --lr 1e-1 --val_freq 50  --drop_freq 500 --sample_freq 25 --m 32

python trainToyOTflow.py --data checkerboard --nt 12 --nt_val 16 --batch_size 10000 --prec double --alph 1.0,15.0,2.0 --niters 20000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32

python trainToyOTflow.py --data swissroll --nt 8 --nt_val 16 --batch_size 5000 --prec double --alph 1.0,30.0,15.0 --niters 5000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32

python trainToyOTflow.py --data circles --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,5.0,1.0 --niters 5000 --lr 5e-2 --val_freq 50  --drop_freq 1000 --sample_freq 25 --m 32

python trainToyOTflow.py --data moons --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,8.0,1.0 --niters 5000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32

python trainToyOTflow.py --data pinwheel --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,30.0,15.0 --niters 5000 --lr 5e-2 --val_freq 50 --drop_freq 1000 --sample_freq 25 --m 32

python trainToyOTflow.py --data 2spirals --nt 8 --nt_val 12 --batch_size 5000 --prec double --alph 1.0,10.0,1.0 --niters 5000 --lr 5e-2 --val_freq 50  --drop_freq 1000 --sample_freq 25 --m 32 
```


### Training and Evaluating Large Data Sets
commands with hyperparameters

```
python trainLargeOTflow.py --data power --niters 36000 --alph 1.0,500.0,5.0 --m 128 --batch_size 10000  --lr 0.03 --nt 10 --nt_val 22 --test_batch_size 120000 --val_freq 30 --weight_decay 0.0 --drop_freq 0

python evaluateLargeOTflow.py --data power --nt 24 --batch_size 120000 --resume yourPowerCheckpt.pth


python trainLargeOTflow.py --data gas --niters 60000 --alph 1.0,1200.0,40.0 --m 350 --batch_size 2000 --drop_freq 0 --lr 0.01 --nt 10 --nt_val 28 --test_batch_size 55000 --val_freq 50 --weight_decay 0.0 --viz_freq 1000 --prec single --early_stopping 20

python evaluateLargeOTflow.py --data gas --nt 30 --batch_size 55000 --resume youGasCheckpt.pth


python trainLargeOTflow.py --data hepmass --niters 40000 --alph 1.0,500.0,40.0 --m 256 --nTh 2 --batch_size 2000 --drop_freq 0 --lr 0.02 --nt 12 --nt_val 24 --test_batch_size 20000 --val_freq 50 --weight_decay 0.0 --viz_freq 500 --prec single --early_stopping 15

python evaluateLargeOTflow.py --data hepmass --nt 24 --batch_size 50000 --resume yourHepmassCheckpt.pth


python trainLargeOTflow.py  --data miniboone --niters 8000 --alph 1.0,100.0,15.0 --batch_size 2000 --nt 6 --nt_val 10 --lr 0.02 --val_freq 20 --drop_freq 0 --weight_decay 0.0 --m 256 --viz_freq 500 --test_batch_size 5000 --early_stopping 15

python evaluateLargeOTflow.py --data miniboone --nt 18  --batch_size 5000 --resume yourMinibooneCheckpt.pth


python trainLargeOTflow.py --data bsds300 --niters 120000 --alph 1.0,2000.0,800.0 --batch_size 300 --nt 14 --nt_val 30 --lr 0.001 --val_freq 100 --drop_freq 0 --weight_decay 0.0 --m 512 --lr_drop 3.3 --viz_freq 500 --test_batch_size 1000 --prec single --early_stopping 15 

python evaluateLargeOTflow.py --data bsds300 --nt 40 --batch_size 10000 --resume yourBSDSCheckpt.pth

```

