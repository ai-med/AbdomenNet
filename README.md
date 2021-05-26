# AbdomenNet - PyTorch implementation

A fully convolutional network for quick and accurate segmentation of abdominal organs
-----------------------------------------------------------
## Getting Started

### Pre-requisites
Please install the required packages for smooth functioning of the tool by running
```
pip install -r requirements.txt
```

### Training your model

```
python run.py --mode=train
```

### Evaluating your model

```
python run.py --mode=eval
```

## Evaluating the model in bulk

Execute the following command for 3-view aggregated evaluations:
```
python run.py --mode=eval_bulk
```
This saves the segmentation files at nifti files in the destination folder.

## Code Authors

* **Jyotirmay Senapati**  - [jyotirmay-senapati](https://www.linkedin.com/in/jyotirmay-senapati/)
* **Anne-Marie Rickmann**  - [annemarierickmann](https://www.linkedin.com/in/annemarierickmann/)


## Help us improve
Let us know if you face any issues. You are always welcome to report new issues and bugs and also suggest further improvements. And if you like our work hit that star button on top. Enjoy :)
