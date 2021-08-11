# OT-Flow
Pytorch implementation of our continuous normalizing flows regularized with optimal transport.

## Associated Publication

OT-Flow: Fast and Accurate Continuous Normalizing Flows via Optimal Transport

Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17113

Supplemental: https://arxiv.org/abs/2006.00104

Please cite as
    
    @inproceedings{onken2021otflow, 
        title={{OT-Flow}: Fast and Accurate Continuous Normalizing Flows via Optimal Transport},
        author={Derek Onken and Samy Wu Fung and Xingjian Li and Lars Ruthotto},
	    volume={35}, 
	    number={10}, 
	    booktitle={AAAI Conference on Artificial Intelligence}, 
	    year={2021}, 
	    month={May},
	    pages={9223--9232},
	    url={https://ojs.aaai.org/index.php/AAAI/article/view/17113}, 
    }

## Set-up

Install all the requirements:
```
pip install -r requirements.txt 
```

For the large data sets, you'll need to download the preprocessed data from Papamakarios's MAF paper found at https://zenodo.org/record/1161203#.XbiVGUVKhgi. Place the data in the data folder. We've done miniboone for you since it's small (and provide a pre-trained miniboone model).

To run some files (e.g. the tests), you may need to add them to the path via
```
export PYTHONPATH="${PYTHONPATH}:."
```

A more in-depth setup is provided in [detailedSetup.md](detailedSetup.md).

## Trace Comparison

Compare our trace with the AD estimation of the trace
```
python compareTrace.py 
```

For Figure 2, we averaged over 20 runs with the following results
```
python src/plotTraceComparison.py 
```



## Toy problems

Train a toy example
```
python trainToyOTflow.py
```

Plot results of a pre-trained example
```
python evaluateToyOTflow.py
```


## Large CNFs

```
python trainLargeOTflow.py
```

Evaluate a pre-trained model
```
python evaluateLargeOTflow.py
```



#### Hyperparameters
Train and Evaluate using our hyperparameters ([see detailedSetup.md](detailedSetup.md))

| Data set           | Train Time Steps | Val Time Steps | Batch Size | Hidden Dim | alpha on C term | alpha on R term | Test Time Steps | Test Batch Size |
|------------------- |----------------- |--------------- |----------- |----------- |---------------- |---------------- |---------------- |---------------- |
| Power              |   10             |        22      |     10,000 |    128     |        500      | 5               | 24              | 120,000         |  
| Gas                |   10             |        24      |     2,000  |    350     |       1,200     | 40              | 30              |  55,000         |
| Hepmass            |   12             |        24      |     2,000  |    256     |        500      | 40              | 24              |  50,000         |
| Miniboone          |   6              |        10      |     2,000  |    256     |        100      | 15              | 18              |    5,000        |
| BSDS300            |   14             |        30      |     300    |    512     |       2,000     | 800             | 40              |   10,000        |
 



### MNIST 

Train an MNIST model
```
python trainMnistOTflow.py
```

Run a pre-trained MNIST
```
python interpMnist.py
```

## Acknowledgements

This material is in part based upon work supported by the US National Science Foundation Grant DMS-1751636, the US AFOSR Grants 20RT0237 and FA9550-18-1-0167, AFOSR
MURI FA9550-18-1-050, and ONR Grant No. N00014-18-1-
2527. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.




