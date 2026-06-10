# Dataset
Dataset used is 'Oxford-IIIT Pet dataset' available at https://www.robots.ox.ac.uk/~vgg/data/pets/

# Traning
Traning this model for 10 epochs on 'Oxford-IIIT Pet dataset' resulted in below evaluation <br>
--- Epoch 10/10 --- <br>
Loss: 0.2050 <br>
Pixel Accuracy: 92.37% <br>
Dice Score (Macro): 0.8660 <br>

# Visual evaluation
There are 2 mask being scaled to visible, actual and predicted of Ragdoll_171 picture.
Scaling is neccesary since value of pixels of masks and predictions are [1,2,3] or [0,1,2] in scale 0-255 (from black to white) so no distinguishable for human eye