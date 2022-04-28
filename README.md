
- The network is available in basic_model.py

- I have converted the RGB images in the dataset into CIELAB colorspace (available in colorize_data.py file) for training and validation, as it allows the intensity and the light filled in the grayscale image to be taken into consideration 
in the form of ‘L’, by adding this conversion part to the ‘colorize_data’ script and thus, solves the problem of adjusting the mood as well.

## Train 
- The training and validation code is in main.py. The input dataset is given in the Dataloaders section at the variable 
'dataset'.
- Download the dataset here: https://drive.google.com/file/d/15jprd8VTdtIQeEtQj6wbRx6seM8j0Rx5/view?usp=sharing 

## Inference
- The path of the testing dataset and the index (number) of the input gray scale image should be given as inputs to the
Trainer.validate() function and the corresponding
color images are stored at 'outputs'.

 
