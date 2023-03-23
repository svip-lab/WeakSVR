# Datasets Configuration Instruction

This project relates to three datasets: *COIN-SV, Diving48-SV, CSV*. The first two are rearranged based on the existing datasets ([COIN](https://coin-dataset.github.io/), [Diving48](http://www.svcl.ucsd.edu/projects/resound/dataset.html)), and the third is newly collected.

| Dataset     | # Tasks | # Videos | # Steps | # Procedures   | # Split Videos     | # Split Pairs      |
| ----------- | ------- | -------- | ------- | -------------- | ------------------ | ------------------ |
| COIN-SV     | 36      | 2114     | 749     | 37 / 268 / 285 | 1221 / 451 / 442   | 21741 / 1000 / 400 |
| Diving48-SV | 1       | 16997    | 24      | 20 / 20 / 8    | 6035 / 7938 / 3024 | 50000 / 1000 / 400 |
| CSV         | 14      | 1941     | 106     | 45 / 25 / -    | 901 / 1039 / -     | 8551 / 1000 / -    |

The following are the download instruction and splits division of above datasets.

---
### COIN-SV
Download: We provide a video id [list](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/COIN-SV/all_ids.txt) accompained with a [script](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/COIN-SV/download_videos.py) to download videos from YouTube

Splits: [train](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/COIN-SV/train_split.txt) / [test](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/COIN-SV/test_split.txt) / [val](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/COIN-SV/val_split.txt)

**NOTE**: We truncate each video according to the start and end timestamps provided in the raw annotation.

### Diving48-SV
Download: [here](http://www.svcl.ucsd.edu/projects/resound/Diving48_rgb.tar.gz)

Splits: [train](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/DIVING48-SV/train_split.txt) / [test](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/DIVING48-SV/test_split.txt) 
/ [val](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/DIVING48-SV/val_split.txt)

### CSV:
Download: [BaiduNetDisk](https://pan.baidu.com/s/1gYYhigjoQjw2OaeZFPwY9g) (extraction code: 9uyk) / [OneDrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/qianych_shanghaitech_edu_cn/EjHfzFTQyWxGuuHsR26u3ncBMYsyiD06foNe4x47-DrfLA?e=cfgL2N)

Splits: [train](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/CSV/train_split.txt) / [test](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/CSV/test_split.txt)

### Get Data Prepared:
1. Download videos through the links above. 
2. Transfer videos to frames (resize to 180 $\times$ 320).
3. Complete the split files with your local data path.

### Something else:
* We also provide video pairs used in our training for reproducing the result reported in the paper if someone interests. ([COIN-SV](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/COIN-SV/train_pairs.txt), [Diving48-SV](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/Diving48-SV/train_pairs.txt), [CSV](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos/blob/main/Datasets/CSV/train_pairs.txt))  
* The labels in COIN-SV and CSV follow the form of **A.B** where **A** indicates the task id and **B** indicates the procedure id in task **A**. The realistic annotation for labels can be found in *label_bank.json* in each dataset's folder.
