<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_uniface_parsing</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_uniface_parsing">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_uniface_parsing">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_uniface_parsing/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_uniface_parsing.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Face parsing (semantic segmentation) using the UniFace BiSeNet model. This algorithm segments face images into 19 different facial components including skin, eyes, nose, mouth, hair, and more.

The BiSeNet model provides accurate face parsing for various applications such as virtual makeup, face editing, and facial analysis.

![Output illustration](https://raw.githubusercontent.com/Ikomia-hub/infer_uniface_parsing/main/images/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_uniface_parsing", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/Ikomia-dev/notebooks/blob/main/examples/img/img_portrait_5.jpg?raw=true")

# Display results
display(algo.get_image_with_mask()) # Visualization overlay
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str, default="resnet18"): Model architecture to use. Available options:
  - "resnet18": Faster inference, good accuracy
  - "resnet34": Better accuracy, slightly slower

**Note**: The algorithm outputs 19 facial component classes:
- 0: Background
- 1: Skin
- 2: Left Eyebrow
- 3: Right Eyebrow
- 4: Left Eye
- 5: Right Eye
- 6: Eye Glasses
- 7: Left Ear
- 8: Right Ear
- 9: Ear Ring
- 10: Nose
- 11: Mouth
- 12: Upper Lip
- 13: Lower Lip
- 14: Neck
- 15: Neck Lace
- 16: Cloth
- 17: Hair
- 18: Hat

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_uniface_parsing", auto_connect=True)

algo.set_parameters({
    "model_name": "resnet34"
})

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/test_images/image0.jpg")

# Display results
display(algo.get_image_with_mask())  # Visualization overlay
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import cv2
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_uniface_parsing", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/yakhyo/uniface/main/assets/test_images/image0.jpg")

# Iterate over outputs
for output in algo.get_outputs()
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
