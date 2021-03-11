# pruning-CNN-with-XAI
This repository provides a CNN pruning for lightening weight using Grad-CAM

# Objective
- CNN lightening that becomes deeper and more complex
- Mininizes loss of accuracy while reducing the computation and time required

# Grad-CAM
![image](https://user-images.githubusercontent.com/72268423/110791659-cde59a00-82b5-11eb-887f-c4393da4078f.png)
 
 CAM needs GAP(Global Average Pool) to extract CAM value, but Grad-CAM can extract Grad-CAM value from all layers. 
 So Grad-CAM can be used for setting up the criterion for pruning
 
 # Design Process
 1. Extract the Grad-CAM value for each filter in each layer,
 2. Set up the appropriate Grad-CAM value for creteria and create mask bit according to creteria
 3. Prune the CNN with mask bits.
 
