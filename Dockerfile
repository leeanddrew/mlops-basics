#FROM huggingface/transformers-pytorch-cpu:latest
FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app

# Copy the locally cached Hugging Face tokenizer/model
COPY ./hf_cache /root/.cache/huggingface

# Set Hugging Face cache environment variable
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION

# aws credentials configuration
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

# install requirements
#RUN yum install git -y && yum -y install gcc-c++
RUN pip install "dvc[s3]==2.8.1"
RUN pip install -r requirements_inference.txt
COPY ./ ./

# initialize dvc
RUN dvc init --no-scm -f

RUN export AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
RUN export AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY

# configuring remote server in dvc
RUN dvc remote add -d model-store s3://models-dvc-mlops-basics/trained_models/

# pulling the trained model
RUN dvc pull models/model.onnx.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
CMD [ "lambda_handler.lambda_handler"]
#EXPOSE 8000
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]