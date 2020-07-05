# Take the base MXNet Container
FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/tensorflow-training:2.0.1-cpu-py3
RUN pip install pandas tensorflow tensorflow_datasets keras utils

# Add Custom stack of code
# RUN git clone https://github.com/ishrat2003/aws_contextual_summary.git

COPY  /code/ /var/code/
WORKDIR /var/code/

ENTRYPOINT ["python", "train.py"]

