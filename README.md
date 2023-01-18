# Segmentation - DIS 
[**Dichotomous Image Segmentation**](https://arxiv.org/abs/2203.03041)<br>
Contrast based highly accurate image segmentation and Mask Generation.

Install Miniconda/Anaconda and Create a virtual environment (Python 3.8) using the command given below:
```
conda create --name <myenv> --python=3.8
conda activate <myenv> 
```
(Linux Installation): In the virtual environment, upgrade pip and install dependencies using:
```
pip install --upgrade pip
pip install -r requirements.txt
```
(Mac Installation): In the virtual environment, Install PyTorch with MPS (GPU) Acceleration
```
# MPS acceleration is available on MacOS 12.3+
conda install pytorch torchvision torchaudio -c pytorch
```

Open the Terminal, Run the following command in terminal
```
zip -s 0 saved_models/model_split.zip  --out saved_models/model.zip
unzip saved_models/model.zip -d saved_models
rm saved_models/*.zip saved_models/*.z01
```

After the Setup, Run the following Command:
```
python run.py [-i INPUT__IMAGE_PATH] [-s SAVE_FILE_PATH]

Ex. python run.py -i 'Downloads/1626897277971.jpg' -s 'Downloads/'

Use: python run.py -h for help

Parsed arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Image Path
  -s SAVE_PATH, --save_path SAVE_PATH
                        Save Path
```