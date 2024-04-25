# MaxQSARing
An autoML toolkit for QSAR Modelling for molecular properties.

## Installation
```
conda env create -n maxqsaring python=3.8.8
conda activate maxqsaring
pip install -r requirements.txt

# Set local data dir:
export DATA_ROOT_DIR={your path of data}
```


## Usage
- Training:
```
python main-v1.py train -tn {task_name} -s {scaffold, random-cv} -d tempdata
```
- `--task_name, -tn`: set task name, eg. herg, bbb_wang
- `--split, -s`: set the split method, eg. random-cv, scaffold, default is scaffold
- `--tmp_dir, -d`: set temp directory where the cache data is saved at.

- Evaluating:
```
python main-v1.py eval -tn {task_name} -d tempdata
```
- `--task_name, -tn`: set task name, eg. herg, bbb_wang
- `--tmp_dir, -d`: set temp directory where the cache data is saved at.


- Predicting:
```
python main-v1.py predict -tn {task_name} -tf {test_path}
```
- `--task_name, -tn`: set task name, eg. herg, bbb_wang
- `--test_file, -tf`: the test 