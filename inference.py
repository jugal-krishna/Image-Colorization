
from main import Trainer # Import main function

# Validate / Test

T = Trainer()  # Instantiate Trainer
T.test(img_dir= 'test_bw', index= 0)  # Convert input to color image here

# Check the folder 'outputs' for output images
