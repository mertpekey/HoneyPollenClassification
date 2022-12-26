## Data Analysis

**Q1-1**

**Pollen Types**

**Is it sufficient to train from scratch:** Number of pollen types are 274 which is so much. In addition, there are a lot of Pollen Types that contains a few images. That means the data we have is not so much to train from scratch but we can give it a try. I think, we should definitely consider transfer learning.

**Is it balanced:** The dataset is not balanced, some classes has a lot more images than others.

**What measures should we take:** We should definitely consider proper data augmentation for this dataset. However, we should be very careful not to corrupt the structure of data.



**Pollen Family**

**Is it sufficient to train from scratch:** Number of pollen families are 47. Some pollen families contains a lot of images but most of them has a few samples compared to them. That means the data we have is not so much to train from scratch but we can give it a try. I think, we should definitely consider transfer learning for this too.

**Is it balanced:** The dataset is not balanced, some classes has a lot of images but some of them has just a few.

**What measures should we take:** We should definitely consider proper data augmentation for this dataset. However, we should be very careful not to corrupt the structure of data.



**Q1-2**

Some classes in the dataset has just a few samples (For the Pollen Types there are minumum 2 but most of them larger than 3 and for the Pollen Family there are 3(min)). We should use data augmentation to increase the number of samples however, we cannot use data augmentation for the validation and test. That's why I thought to use %50 train, %25 validation, %25 test which are (2 train, 1 validation, 1 test). We should definitely consider cross validation because we should validate our model with different validation datasets (because otherwise there were just a little bit validation data) to make our model more robust.

I may think to get rid of classes which has lower than 4 images. With the minimum 4 images, we can set a test dataset and can use others with cross validation. (However, we can do 2-fold cross validation maximum. If we get rid of lower than 6, we can do more.)



**Q1-3**

We should definitely do normalization. Pixel values should be set between 0 and 1 instead of 0 and 255. In addition, we should calculate mean and standard deviation of the dataset for the 3 channels to normalize the dataset.



**Q1-4**

We should definitely use data augmentation. We have just a few samples for most classes. I think, flips (Horizontal, Vertical) and rotation does not corrupt the structure of data. In addition, we can change the blur and change the contrast or color (I am not sure about this). I will add some others after I research more.



### Folder Structure ###

- model_checkpoints/
    - will contain model checkpoints

- notebooks/
    - data_analysis.ipynb - **Dataset Class and analysis of dataset**
    - training_notebook.ipynb - **Training of models will be tested here**

- src/
    - model/trainer.py - **Contains trainer class**
    - model/config.py - **Contains configs (May be separated as model_configs and configs)**
    - model/model_classes.py - **Will contain all the model classes (May be separated)**
    - data/get_pollen_family.py - **Get pollen Families given pollen types**
    - data/make_dataset.py - **Contains codes about dataset class**
- utils/eval_utils.py - **Utilization functions for model evaluation**
- utils/eval_utils.py - **All the other functions**



### TO-DO ###

- Dataset Class **(Dataset created in a Dataframe with Pollen Type and Family)**
    - Data Analysis **(Plots not fine. Answers grammer should be corrected. Cross validation, Data augmentation and train/val/test split size can be re-checked.)**
- Model
    - Folder contains models **(model/model_classes.py)**
    - Checkpoint folder
    - optimization_funcs.py
    - loss_funcs.py
    - util_functions.py
        - view_dataloader_imgs **(Done)**
        - create_dataset - create_dataloader **Done**
        - plot_loss - plot_acc - tensorboard?
- Argument Parser