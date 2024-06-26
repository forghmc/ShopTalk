# End-to-end-Machine-Learning-Project-with-MLflow


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
##STEPS:

### Clone the repository

```bash
https://github.com/forghmc/ShopTalk.git

```
Download following files from gdrive - https://drive.google.com/drive/folders/14BcyjB7wbvvgy9g3_suVVey4yufOfz24

1. processed_dataset_target_data_with_captions_only.csv
2. resize.zip -keep resize folder in parent directory and unzip it.

### Update Env Variables

Goto  setenv.sh  in parent folder edit the paths and keys make sure the secret keys are set and csv path should be same as the downloaded file is kept.

After setting variables Run setenv.sh if needed gives executable permission 
```bash
chmod +x setenv.sh
```
```bash
sh setenv.sh
```
### Create conda env
```bash
conda create -p ./env python=3.10 -y
```
### Activate env 
```bash
conda activate ./env
```


### install the requirements for backend

```bash
pip install -r generate_request/requirements.txt' 

```


## Running Build

# Resize Image - this is neeeded once
1. Provide the value for extracted image folder with in file 
2. run utils/image_exctract.py
# To start streamlit app
1. Make sure your path to image  in appv2.py are pointing towards extracted resized image folder
2. Update the path the 'processed_dataset_target_data_with_captions_only.csv' in appv2 file.
3. Set pine cone key in env with "export pkey=<your key>" and also update index_name.

3. Then 
```bash
streamlit run appv2.py
```
```bash
python -m streamlit run appv2.py

```
	
# To Kill streamlit process
```bash
ps -ef | grep streamlit | grep -v grep | awk '{print $2}' | xargs kill
```
Return to your normal shell by typing:
```bash
 deactivate
```
## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/someshnaman/End_to_end_MLOPS_project.mlflow \
MLFLOW_TRACKING_USERNAME=someshnaman \
MLFLOW_TRACKING_PASSWORD=6e7e6b4e21fb207c4cbf0d4d7f20506e23e748cc \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/someshnaman/End_to_end_MLOPS_project.mlflow

export MLFLOW_TRACKING_USERNAME=someshnaman 

export MLFLOW_TRACKING_PASSWORD=6e7e6b4e21fb207c4cbf0d4d7f20506e23e748cc

```
# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
 	- Save the URI: 023110776410.dkr.ecr.us-east-1.amazonaws.com/shoptalk
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:

## 6. To start streamlit app
```bash
streamlit run app.py
```
	
## 7. To Kill streamlit process
```bash
ps -ef | grep streamlit | grep -v grep | awk '{print $2}' | xargs kill
```
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model
