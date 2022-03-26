# Detectron2 Object Detection with TTA
## 1. Build image and run docker container
`docker-compose up --build -d && docker exec -it detectron2_header_1.0v bash`.

## 2. Set TTA's Hyperparameters
In `run.py`and `tta.py` file, you can change Hyperparameters of TTA.
### 2.1. no TTA
`args.TTA` must be `False`.
### 2.2. horizontal flip
**Flip the image horizontally.**<br>
In `run.py`, Hyperparameter `args.TTA` must be `True`. <br>
In `tta.py`, Hyperparameter `_flip` set to `True`.

### 2.3. multi scale
**Resize the given size.<br>
Number of input image for TTA are 10 => (9 images are resized, 1 image is original size)**<br>
In `run.py`, Hyperparameter `args.TTA` must be `True`. <br>
In `tta.py`, Hyperparameter `_multi_scale` set by `list`.

### 2.4. contrast
In `run.py`, Hyperparameter `args.TTA` must be `True`. <br>
In `tta.py`, Hyperparameter `_contrast` set to `True`.

**Contrast intensity is uniformly sampled in (intensity_min, intensity_max).<br>
    - intensity < 1 will reduce contrast <br>
    - intensity = 1 will preserve the input image <br>
    - intensity > 1 will increase contrast**<br>

## 3. Just run code
After setting TTA parameters. <br>
Run `python run.py`.

## 4. check the test AP in your terminal.
Or check a log which located in `/home/src/model/log.txt`.

## 5. Best Result(default setting)
`flip` = True <br>
`multi_scale` = [400, 600, 800, 1000] <br>
`contrast` = [ ]<br>
**baseline AP: 40.2161**<br>
**best AP: 41.6996**
