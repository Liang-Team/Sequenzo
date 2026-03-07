# 生成的数据集目录

此目录存放由 `random-sequences-generation/generation.py` 生成的数据集。

每个子目录对应一个数据集（由 `run_test.sh` 命名为 `n{n}_l{l}_d{d}_it{it}`），包含：
- `output1.dat`, `output2.dat`, `output3.dat`, `output4.dat`：各聚类对应的序列数据
- `output1.pat`, `output2.pat`, ...：隐藏模式描述（如有）

运行脚本时使用 `-d <数据集名>` 指定读取的数据集，例如：
```bash
python run_tanat.py -c 4 -m edit -d n100_l8_d20_it1
python run_sequenzo.py -c 4 -d n100_l8_d20_it1
Rscript run_traminer_clustering.R -c 4 -d n100_l8_d20_it1
```
