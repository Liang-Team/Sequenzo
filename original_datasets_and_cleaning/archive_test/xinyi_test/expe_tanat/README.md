


### Installation

```{bash}
python -m venv venv
source venv/bin.activate

git clone git@gitlab.inria.fr:tguyet/random-sequences-generation.git
cd random-sequences-generation
pip install -e .
cd ..

pip -r requirements.txt
```

Install R and its TraMineR package and optparser packages
```
install.packages(['TraMineR', 'optparser'])
```
