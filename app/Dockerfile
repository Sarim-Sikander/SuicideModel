FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN python3 -m ensurepip
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_lg

ADD src ${LAMBDA_TASK_ROOT}

CMD ["main.handler"]