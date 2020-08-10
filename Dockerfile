FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:2.0.1-cpu-py3
#RUN pip install pandas tensorflow tensorflow_datasets keras utils logging

RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip uninstall -y awscli boto3 botocore 
RUN pip install awscli --upgrade --use-feature=2020-resolver
RUN pip install botocore --upgrade --use-feature=2020-resolver
RUN pip install boto3 --upgrade --use-feature=2020-resolver
RUN pip install nltk --use-feature=2020-resolver
RUN python -W ignore -m nltk.downloader punkt
RUN python -W ignore -m nltk.downloader averaged_perceptron_tagger

COPY  /code/ /opt/ml/
WORKDIR /opt/ml/
RUN pip install -r requirements.txt --use-feature=2020-resolver


ENV PYTHONUNBUFFERED=1
# Add Custom stack of code
# RUN git clone https://github.com/ishrat2003/aws_contextual_summary.git



ENTRYPOINT ["python", "train.py"]

