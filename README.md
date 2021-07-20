# Reinforcement Learning with Logical Action-Aware Features for Polymer Discovery

environments, data and experiments for Reinforcement Learning with Logical Action-Aware Features for Polymer Discovery paper

### Install material discovery environments
```bash
pip install -e md-envs
```

### Extract dataset from zip file
```bash
cd data/
unzip polymerDiscovery.zip 
cd ..
```

### Update pickled function in dataset

```bash
python scripts/update_pickled_function.py
```

### Test with pretrained model

```bash
python scripts/main.py test -f 'logical_with_regressor' -a 'dqn' -m 'models/DQN_logical_with_regressor.model'
```

### Train a model

```bash
python scripts/main.py train -f 'logical' -a 'dqn' -o 'models/DQN_logical.model'
```

### Usage

```bash
python scripts/main.py train -h

python scripts/main.py test -h
```