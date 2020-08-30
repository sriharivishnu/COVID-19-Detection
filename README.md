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

#### Close to 93%-97% accuracy
(For the 15 datasets used as evaluation)

<img width="618" alt="Screen Shot 2020-08-30 at 1 13 09 PM" src="https://user-images.githubusercontent.com/37857112/91665316-b043a680-eac2-11ea-8e5f-81d271e4f15c.png">

(Last line is the test results)


<img width="711" alt="Screen Shot 2020-08-30 at 1 15 20 PM" src="https://user-images.githubusercontent.com/37857112/91665338-dcf7be00-eac2-11ea-8fda-8a891e00dd3f.png">

#### Sample Test Accuracy

<img width="936" alt="Screen Shot 2020-08-30 at 1 13 18 PM" src="https://user-images.githubusercontent.com/37857112/91665318-b33e9700-eac2-11ea-9720-687218b6f1a1.png">

## Summary
While the CT scan dataset offers a lot more data, the Chest X Ray dataset consistently provided good results, despite the quantity of data. This works on this dataset, however, this should not be used as a Diagnostic tool, and simply serves as a proof of concept. One of the reasons behind this is the limited dataset available of lung X-Rays; a better, diverse data set would offer more generality.

### Datasets Obtained From
- https://github.com/UCSD-AI4H/COVID-CT
- https://github.com/ieee8023/covid-chestxray-dataset
- https://www.kaggle.com/bachrr/covid-chest-xray?select=images
