# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

### PROJECT INTRODUCTION
The project name is: Image classification using AWS sagemaker. This project is using AWS sagemaker to finetune a pretrained model that can perform image classification
on dog breed classification dataset found here  to classify between different breeds of dogs in images. The dataset contains images from 133 dog breeds divided into training, testing and validation datasets.

This project is devided on 3 main tasks:

1- Data preparation, Fetch and upload the data to AWS S3.

2- Training and include the following steps:

       1. finetuning a pretrainded model (ResNet50 model is the used model) to find the best hyperparameters.
       2. Using the best hyperparameters and train and finetune a new model and monitor its performance using model debugging and profiling.
       
3- Deploy the optimized model to an endpoint and testing it with a sample image and get a prediction.

The project summarize is givin in the following figure:

### Project Setup Instructions
Setting up the environment for building the project and preparing your data for training your models as follow:

1.Set up AWS by opening it through the classroom and open sagemaker studio and create a folder for the project
2. Download the Starter Files by cloning Github RIPO (https://github.com/udacity/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter) and uploading starter files to workspace.
3. Preparing data by downloading dog breed classification data and uploading it to S3 bucket.
4. Installing the necessary packages  for this project's execution:smdebug, jinja2, sagemaker,boto3, sagemaker.tuner, sagemaker.pytorch, sagemaker.debugger





## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Apretrained model (Resnet50 Model) is used for image classification and this network is trained for classifying images into one of the 1000 categories or classes. 
The parameters that used to tune the model are: learning rate, epochs and batch-size. 
hyperparameter_ranges= {"lr": ContinuousParameter(0.001, 0.1),
                        "epochs":IntegerParameter(1,4),
                        "batch-size":CategoricalParameter([32, 64, 128, 256, 512])}
                        

Remember that your README should:
- Include a screenshot of completed training jobs
![training_jobs](https://user-images.githubusercontent.com/81697137/226155910-e816f388-725f-40fe-9ac6-bbc28c7fbf85.png)

- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs
- ![HP_tunning_job](https://user-images.githubusercontent.com/81697137/226155945-86d2d02b-1fbb-4c69-af91-d00ecdd00271.png)
- ![HP_tunning_job1](https://user-images.githubusercontent.com/81697137/226157196-c43b5703-7294-4afe-8612-906d3e2d7837.png)




## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
Model debugging and profiling are performed using 6 steps:
 1. Import SMDebug framework class.
 2. Set the SMDebug hook for the test phase.
 3. Set the SMDebug hook for the training phase.
 4. Set the SMDebug hook for the validation phase.
 5. Register the SMDebug hook to save output tensors.
 6. Pass the SMDebug hook to the train and test functions.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
![debug_profileresults](https://user-images.githubusercontent.com/81697137/226156234-249ccf25-8e8d-43b4-a3ee-51282dd5a75c.png)


**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

An inference script is created to implements the following functions to get a prediction; Net function, the model_fn function(calls the loaded model saved after retraining hpo.py script using finetuned parameters), input_fn function (process the image/url uploaded to the endpoint) and predict_fn function.

The instructions on how to query the endpoint is as follow:
Read a random image given by url or uploaded from dogImages/test/ directory and call the predict method of our predictor with the input image. We can then parse the result for the answer. Url and an uploaded image are used to query the endpoint and sample input of uploaded image is given below:
![image](https://user-images.githubusercontent.com/81697137/226215650-08218eb0-c548-4a86-8fac-93432c78c322.png)

The result of quering the endpoint was correct (the answer was 51) which refer to 051.Chow_chow images.
 
 

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![active_endpoint](https://user-images.githubusercontent.com/81697137/226156285-75901835-f9b7-4d99-bf95-43781c15db6a.png)


## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
