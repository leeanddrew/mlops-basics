FROM public.ecr.aws/lambda/python:3.9

# Cache HF models and set env
COPY ./hf_cache /root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# AWS credentials
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

# Create model dir
ARG MODEL_DIR=./models
RUN mkdir -p $MODEL_DIR
ENV TRANSFORMERS_CACHE=$MODEL_DIR
ENV TRANSFORMERS_VERBOSITY=error

# System requirements (Ubuntu-based image)
RUN yum install -y git gcc-c++ make && yum clean all

# Copy source
COPY . /var/task/

# Install dependencies
RUN pip install "fsspec==2021.10.0" "s3fs==2021.10.0" "dvc[s3]==2.8.1"
RUN pip install -r requirements_inference.txt
ENV PYTHONPATH="${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Initialize and pull DVC-tracked model
RUN dvc init --no-scm -f
RUN dvc remote add -d model-store s3://models-dvc-mlops-basics/trained_models/
RUN dvc pull models/model.onnx.dvc

# Final step for Lambda
RUN chmod -R 0755 $MODEL_DIR
CMD [ "lambda_handler.lambda_handler" ]
