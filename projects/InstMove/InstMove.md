## InstMove





# InstMove: Instance Motion for Object-centric Video Segmentation

This is the official implementation of the paper : "[InstMove: Instance Motion for Object-centric Video Segmentation](https://arxiv.org/abs/xxxxxxxx)".



## Introduction

In this work, we study the instance-level motion and present InstMove, which stands for **Inst**ance **M**otion for **O**bject-centric **V**ideo S**e**gmentation. In comparison to pixel-wise motion (optical flow), InstMove mainly relies on instance-level motion information that is free from image feature embeddings, and features physical interpretations, making it more accurate and robust toward occlusion and fast-moving objects. To better fit in with the video segmentation tasks, InstMove uses instance masks to model the physical presence of an object and learns the dynamic model through a memory network to predict its position and shape in the next frame. With only a few lines of code, InstMove can be integrated into current SOTA methods for three different video segmentation tasks and boost their performance. Specifically, we significantly improve the previous arts by 1.5 AP on OVIS dataset, which features heavy occlusions, and 4.9 AP on YouTubeVIS-Long dataset, which mainly contains fast moving objects. These results suggest that instance-level motion is robust and accurate, and hence serving as a powerful solution in complex scenarios for object-centric video segmentation.

### About code

Working on to make it readable, will be ready in a few weeks.
