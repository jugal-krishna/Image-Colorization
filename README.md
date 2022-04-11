## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)

- I can improve the overall image quality for the above system using a higher level model like a RESNET-50 or RESNET-101. 

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)
- I have I have converted the RGB images in the dataset into CIELAB colorspace for training and validation because 
- it allows the intensity and the light filled in the grayscale image to be taken into consideration 
- in the form of ‘L’, by adding this conversion part to the ‘colorize_data’ script and thus, solves the problem of adjusting the mood as well.

## Train 
- The training and validation code is in main.py. The input dataset is given in the Dataloaders section at the variable 
- 'dataset'.

## Inference
- The path of the testing dataset and the index (number) of the input gray scale image should be given as inputs to the
Trainer.validate() function and the corresponding
color images are stored at 'outputs'.

 
