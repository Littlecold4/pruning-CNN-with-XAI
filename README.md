# pruning-CNN-with-XAI
This repository provides a CNN pruning for lightening weight using Grad-CAM.

# Objective
- Lightening weight CNN that becomes deeper and more complex.
- Reduce the computation and time required while minimizing the loss of accuarcy.

# Grad-CAM
![image](https://user-images.githubusercontent.com/72268423/110791659-cde59a00-82b5-11eb-887f-c4393da4078f.png)
 
 CAM needs GAP(Global Average Pool) to extract CAM value, but Grad-CAM can extract Grad-CAM value from all layers. 
 So Grad-CAM can be used for setting up the criteria for pruning.
 
 # Design Process
 1. Extract the Grad-CAM value for each filter in each layer.
 2. Set up the appropriate Grad-CAM value for criteria and create mask bit according to criteria
 3. Prune the CNN with mask bits.
 
# References
[1]Gunning, D., & Aha, D. (2019). DARPA’s Explainable Artificial Intelligence (XAI) Program. AI Magazine, 40(2), 44-58.
 https://doi.org/10.1609/aimag.v40i2.2850
 

[2] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning deep features for discriminative localization. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2921-2929)
https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Zhou_Learning_Deep_Features_CVPR_2016_paper.html


[3] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision (pp. 618-626).https://arxiv.org/abs/1610.02391

[4] ] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778!https://arxiv.org/abs/1512.0338
