# Image retrieval with distractors

**by Andrii Shevtsov**

Repository of experiments for the [UCU Computer Vision 2024](https://www.kaggle.com/competitions/ucu-computer-vision-2024) Kaggle competition.

## Environment

We are using conda to manage the environment here. You can install it using the [guide](https://docs.anaconda.com/anaconda/install/index.html).

### Creating the environment

```bash
conda env create -f environment.yaml
```

### Updating the environment.yaml file (dev)

On Linux:

```bash
conda env export | grep -v "^prefix: " > environment.yaml
```

On Windows:

```bash
conda env export | findstr /V "^prefix: " > environment.yml
```

## Data

You can download data on this [Kaggle page](https://www.kaggle.com/competitions/ucu-computer-vision-2024/data) with "Download All" button and unzip it in the `data` directory here.

## Experiment log

