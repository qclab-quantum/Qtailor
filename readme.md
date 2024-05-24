
# AI-Powered Algorithm-Centric Quantum Processor Topology Design

This repository is the official implementation of ***AI-Powered Algorithm-Centric Quantum Processor Topology Design.***

![Overview](./temp/overview.png)

## Requirements

We recommend **Anaconda** as Python environment manager

 Install Pytorch on Linux:

```setup
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Guide of install Pytorch on windows or with GPU  is available at [Pytorch Get Started](https://pytorch.org/get-started/locally/)

To install requirements:

```setup
pip install -r requirements.txt
```

## Configuration

you have the option to utilize the default settings or customize the important items in the config.xml :

```
# traing iterations
iters_arr
  - 5
  
# relative path of  store in benchmark folder
circuits:
  -qft\\qft_indep_qiskit_5.qasm
  
# about 0.5x the number of CPU cores, for laptop you can set to 4
num_rollout_workers: 4
```



## Training

To train the model(s) in the paper, run this command:

```train
python rllib_run.py
```

>📋  When training complete the training result will be automatically saved in benchmark/a-result/xxx.csv
>
>The column 'rl' means the depth of mapped circuits results from our method, the column 'results'  results represents the lower left corner of the matrix , as we mention in **Section 2.1 (line 92)**

## Evaluation

To evaluate the result,

1. open utils/benchmark.py  and modify the main function

```python
if __name__ == '__main__':

    array  = []

    qasm = 'portfolio_vqe/portfoliovqe_indep_qiskit_10.qasm'

    Benchmark.compare_gates(qasm=qasm,array=array,bits = 10)
```

2. replace the existing values with those retrieved from the CSV file.

![](./temp/readme2.png)

![](./temp/readme1.png)



3. run the main Function in your editor , or  run:

   ```shell
   python utils/benchmark.py
   ```



## Results

Our model achieves the following performance on circuits depth :

![](./data/fig/benchmarkBar.png)
