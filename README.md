# ViT Model From scratch
 This is repository where we replicated ViT model only by using basic function from Pytorch.
This is not exact replication from: https://arxiv.org/abs/2010.11929 We are missing few features and it is modified to run on local computer (1070Ti 8GB at least)
How ever the basic logic behind this is correct and you can see the minimum that is requiree to run this model. 

This repository contains main file and moduls folder (where are all the extra files that we use).
The ViT model is contained in model_builder.py

In main function you can train the custom model with random weights not trained before (however i do not recommed unless you have large dataset) and you can also run the transfer learning that performes way better as expected. 

We used the dataset from ZTM course from Mrdbourke: https://github.com/mrdbourke/pytorch-deep-learning/tree/main/data

Also this code is from learning in ZTM course but it is modified to run on local machine.

Any additional question or information you can contac me on Discrod: Edesak#5182
