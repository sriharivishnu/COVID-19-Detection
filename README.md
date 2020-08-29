# COVID-19-Detection
Covid-19 Detection in medical imaging.

## ct_scan_model.py
Trained a CNN (Convolutional Neural Network) to recognize Covid-19 in Lung CT scans. The dataset was adequate with over 700 entries, however, the quality of the images made it difficult for the model to converge properly.
Averages around 70-75% Accuracy.

**Obtained an accuracy of 76.7% on the Test Data.**

<img width="646" alt="Screen Shot 2020-08-29 at 12 41 29 PM" src="https://user-images.githubusercontent.com/37857112/91641832-f6cacf80-e9f4-11ea-9c7e-18bdf3942350.png">
(The last line is the test results)

#### Obtained 80.0% Accuracy

<img width="636" alt="Screen Shot 2020-08-29 at 2 02 17 PM" src="https://user-images.githubusercontent.com/37857112/91643333-537fb780-ea00-11ea-91b8-189468bcbecf.png">

# Can we do better?

## x_ray_model.py
Trained a CNN to recognize Covid-19 in Lung X-Rays. The dataset was much higher quality, and easier to isolate features. However, there were much less data available for training, but the results were good.

#### Obtained 90.0% Accuracy

<img width="623" alt="Screen Shot 2020-08-29 at 2 53 13 PM" src="https://user-images.githubusercontent.com/37857112/91644150-65b12400-ea07-11ea-9fad-5f18784f5af7.png">
(The last line is the test results)

<img width="731" alt="Screen Shot 2020-08-29 at 2 56 30 PM" src="https://user-images.githubusercontent.com/37857112/91644187-d3f5e680-ea07-11ea-999c-a80e717aa7b6.png">

#### Even 100% on Convergence
(For the 10 datasets used as evaluation)

<img width="642" alt="Screen Shot 2020-08-29 at 3 29 28 PM" src="https://user-images.githubusercontent.com/37857112/91644693-8fb91500-ea0c-11ea-9e3a-91da974e0dd0.png">

(Last line is the test results)

<img width="705" alt="Screen Shot 2020-08-29 at 3 29 47 PM" src="https://user-images.githubusercontent.com/37857112/91644695-921b6f00-ea0c-11ea-87bb-1c2a38edfa7c.png">

While 100% accuracy seems unrealistic, the model was predicting the correct value for 15 consecutive test data (0.0003%) chance of happening, (especially multiple times in a row). 

#### Images that it predicted correctly
<img width="808" alt="Screen Shot 2020-08-29 at 3 30 11 PM" src="https://user-images.githubusercontent.com/37857112/91644696-92b40580-ea0c-11ea-99fe-fdb01acd6ead.png">



## Summary
While the CT scan dataset offers a lot more data, the Chest X Ray dataset consistently provided good results, despite the quantity of data.

### Datasets Obtained From
https://github.com/UCSD-AI4H/COVID-CT
https://github.com/ieee8023/covid-chestxray-dataset
https://www.kaggle.com/bachrr/covid-chest-xray?select=images
