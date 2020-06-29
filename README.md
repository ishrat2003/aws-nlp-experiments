# aws-nlp-experiments
AWS Sagemaker nlp research experiments 


# Setting up local sagemaker

Local amazon sagemaker docker instance is created using [sagemaker-notebook-container](https://github.com/qtangs/sagemaker-notebook-container). Follow the following steps for setup.
- Install [docker](https://docs.docker.com/docker-for-mac/install/).
- Install AWS CLI as follows:
```sh
$ curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
$ sudo installer -pkg AWSCLIV2.pkg -target /
```
- Verify AWS CLI installation as follows
```sh
$ which aws
$ aws --version
```
- Create a new [IAM](https://console.aws.amazon.com/iam/) user .
- Get the and store the access key and secret key in Roboform.
- Set default configuration or use profile configuration.
```sh
$ aws configure --profile ishrat_experiments
AWS Access Key ID [None]:  YOUR_ACCESS_KEY
AWS Secret Access Key [None]: YOUR_SECRET_KEY
Default region name [None]: eu-west-1
Default output format [None]: json
```
- Update AWS_PROFILE in docker-compose.yml as required
- Do the following to start SageMaker notebook instance
```sh
$ docker-compose up
```
- Copy the Jupyter link from the console with token and open it in browser.