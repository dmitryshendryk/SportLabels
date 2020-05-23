# SportLabels

## Classify Size and Boxes

Requirements is to sort the Size and images and Box images into seperate folders using computer vision.

### Train 

```python
python main.py  train --network=imagenet --device=cpu --save_weights=./imagenet/weights/model_final.pth
```
### Visualize

```python
python main.py  visualize --network=imagenet --device=cpu --weights=./imagenet/weights/model_final.pth
```
### Start Pipeline

```python
python main.py  process  --folder_path=/Users/dmitry/Documents/Business/Projects/Upwork/SportLabels/code/imagenet/data/test  --device=cpu --save_weights=./imagenet/weights/model_final.pth
```
