# LT2326/LT2926 WikiArt peregrinations  - Daniilidou Viktoria Paraskevi
In order to run the files you need python3 and then the name of the file. 
  # Bonus A - Make the in-class example actually learn something

  For this part you will need to run train.py and test.py. 
   
  I observed that the reason why there was so low accuracy -The initial accuracy was around 0.02- in the initial model is the different order of the training and testing classes. So, in order to solve this problem I defined in the beginning a sorted list fixed_classes that the model is going to use both in training and testing. Below you will find the initial training and testing classes and the fixed_classes that the model uses now in order to improve the accuracy. 


Training classes: ['Early_Renaissance', 'Naive_Art_Primitivism', 'Impressionism', 'Mannerism_Late_Renaissance', 'Pointillism', 'Baroque', 'Symbolism', 'Synthetic_Cubism', 'Action_painting', 'Art_Nouveau_Modern', 'Rococo', 'Analytical_Cubism', 'Expressionism', 'Realism', 'Romanticism', 'High_Renaissance', 'Pop_Art', 'Post_Impressionism', 'Contemporary_Realism', 'Color_Field_Painting', 'Minimalism', 'New_Realism', 'Abstract_Expressionism', 'Northern_Renaissance', 'Ukiyo_e', 'Cubism', 'Fauvism'] 

Testing classes: ['High_Renaissance', 'Rococo', 'Baroque', 'Early_Renaissance', 'Expressionism', 'Romanticism', 'New_Realism', 'Action_painting', 'Pop_Art', 'Naive_Art_Primitivism', 'Symbolism', 'Analytical_Cubism', 'Mannerism_Late_Renaissance', 'Abstract_Expressionism', 'Color_Field_Painting', 'Ukiyo_e', 'Fauvism', 'Contemporary_Realism', 'Minimalism', 'Cubism', 'Art_Nouveau_Modern', 'Pointillism', 'Northern_Renaissance', 'Synthetic_Cubism', 'Realism', 'Impressionism', 'Post_Impressionism']

![image](https://github.com/user-attachments/assets/12bcac01-e06b-40b0-a77e-f21215597b7e)




Except for fixing the order of the classes, I also added a dropout=0.5 to prevent overfitting and the final accuracy now is 0.19365079700946808

# Part 1 - Fix class imbalance

The data has very unbalanced classes which may lead to problematic performance. For example, Impressionism has 2269 samples while Analytical_Cubism has only 15 samples.

![image](https://github.com/user-attachments/assets/76a80a5e-4157-4eac-a999-8dee2f95e3aa)

Upsampling from scikit learn was used (sklearn.utils.resample) in order to resample the sparse samples in a consistent way and deal with class imbalance. 
So, in the final training dataset after upsampling each class has 2269 samples. 
After running the upsampled dataset the accuracy is 0.17777778208255768. So, while better accuracy was explected, probably due to the highly imbalanced dataset naive oversampling might mitigate class imbalance effects by simply duplicating minority class examples and lead to overfitting

# Part 2 - Autoencode and cluster representations 

For this part you will need to run encodertraining.py and encodertest.py. 

   

