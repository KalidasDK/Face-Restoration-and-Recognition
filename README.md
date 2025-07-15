# Face-Restoration-and-Recognition
This project was made by using the GFPGAN model for face restoration published [here](https://github.com/TencentARC/GFPGAN) by TencentARC and combining its results with a face recognition model pusblished [here](https://github.com/ageitgey/face_recognition) by Adam Ageitgey. 

Its working involves uploading a degraded facial image to the GFPGAN module as input. It restores the faces within the input and passes this restored image to the face recognition module. This second module identifies the people in the image by comparing their facial encodings to the encodings stored within the 'known_people' folder. 

I used the faces of my classmates (with their consent, of course) in the 'known_people' folder to test this project, and it worked quite well.

To try it out, you can use the following instructions:

#### Please note: 
I ran this in Python 3.10.16 on my M1 pro Macbook Pro.
Also, I strongly recommend that you use a conda environment while working on this.

### Step 1: Clone the GFPGAN repository
```
git clone https://github.com/TencentARC/GFPGAN.git
```
### Step 2: Make the following changes to fix the outdated packages in GFPGAN
Go to the folder having your conda environment's python libraries. There, head to ```lib/ python 3.10/ site-packages/ basicsr```.
Then, open ```data/ degradations.py``` and change line 8 to ```
from torchvision.transforms.functional import rgb_to_grayscale```. This should help it run smoothly.
### Step 3: Installing face_recognition
Follow the steps given in the installation section [here](https://github.com/ageitgey/face_recognition) for your particular OS.

### Step 4: Running the project
Run the ```app.py``` file to see the results.

I hope you find this useful. 

## Thanks
1. I am indebted to the authors of the paper [GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior](https://arxiv.org/pdf/2101.04061) for their work that I have used here. I've used their pre-trained weights too.
2. My sincere thanks to Adam Ageitgey [@ageitgey](https://x.com/ageitgey). I learned a lot about face recognition from his [article](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) for which I am grateful.
