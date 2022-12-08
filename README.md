# UniParser pytorch  
This is an unofficial pytorch implement for paper: 
'UniParser: A Unified Log Parser for Heterogeneous Log Data'

If you feel helpful for your work, please leave your Stars.
### Code organization:
+ [benchmark](benchmark): UniParser_benchmark scripts to reproduce the evaluation results.
+ [logparser](logparser): the logparser package, including the model file and other scripts.
+ [logs](logs): Datasets and manually parsed structured logs with their templates (ground truth) from [LogHub](https://github.com/logpai/loghub).
 
### Environments: 
Here are the main packages

Numpy   1.21.4

PyTorch 1.10 

Pandas  1.3.4

### Get Start:
Run UniParser_benchmark.py in the [benchmark](benchmark) folder. 

Results are recorded in the ./benchmark/UniparserResult/ folder in the format of (precision, recall, f1, pa, mla_sc)





