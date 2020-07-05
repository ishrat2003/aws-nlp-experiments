# aws-nlp-experiments
AWS Sagemaker nlp research experiments.


## Setup AWS CLI

Follow the following steps for setting a local environment for running training.

1. Install [docker](https://docs.docker.com/docker-for-mac/install/).
2. Install AWS CLI as follows:
```sh
$ curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
$ sudo installer -pkg AWSCLIV2.pkg -target /
```
3. Verify AWS CLI installation as follows
```sh
$ which aws
$ aws --version
```
4. Create a new [IAM](https://console.aws.amazon.com/iam/) user and a role 'isSageMaker'. The role should have following permissions.
```
AmazonEC2FullAccess
AmazonEKSClusterPolicy
AmazonEKSWorkerNodePolicy
AmazonEC2ContainerRegistryFullAccess
AmazonS3FullAccess
AmazonEKSServicePolicy
AmazonSageMakerFullAccess
```

5. Get and store the access key and secret key in Roboform.
6. Set default configuration or use profile or both. configurations.
```sh
$ aws configure --profile ishrat_experiments
AWS Access Key ID [None]:  YOUR_ACCESS_KEY
AWS Secret Access Key [None]: YOUR_SECRET_KEY
Default region name [None]: eu-west-1
Default output format [None]: json
```
7. Set AWS_PROFILE environment variable
```sh
export AWS_PROFILE=ishrat_experiments
```

8. Check default and profile credential setup in 
 ~/.aws/credentials

## Setup Jupyter notebook to run training job

1. Install [Anaconda](https://www.anaconda.com/products/individual)
2.  After installing Anaconda check jupyter
```sh
which jupyter
```
It should point to .../anaconda3/bin/jupyter

3. Create local Conda environment and register it with jupyter kernel
```sh
conda update --all --yes
conda create -n localsm python==3.7
conda activate localsm
conda install pip pandas tensorflow tensorflow_datasets keras utils s3fs, sagemaker
conda install ipykernel
python3 -m ipykernel install --user --name localsm --display-name "Python (localsm)
```
4. Install packages for default python3 environment
```sh
pip install pandas tensorflow tensorflow_datasets keras utils s3fs, sagemaker
```
5. Add alias in bash profile.
```sh
alias jupyter=/opt/anaconda3/bin/jupyter
```
6. Start jupyter in this directory
```sh
jupyter notebook&
```

## Processing training image

1. To build training image use
```sh
./build.sh
```
2. To push traing image to AWS ECR use
```sh
./push.sh
```
3. To test training job localy use
```sh
./local/train.sh
```

## Additional notes

- Get the list of iam roles for traing and copy the arn and put it in notebook.
```sh
aws iam list-roles --profile ishrat_experiments | grep isSageMaker
```


##  References

- https://github.com/aws/sagemaker-training-toolkit

- https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb

- https://github.com/qtangs/sagemaker-notebook-container


