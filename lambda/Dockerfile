FROM public.ecr.aws/lambda/python:3.12

# Copy project files needed for pip install
COPY pyproject.toml README.md ${LAMBDA_TASK_ROOT}/
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# Install imgeda and all runtime dependencies, then clean up source
RUN pip install ${LAMBDA_TASK_ROOT}/ --no-cache-dir \
    && rm -rf ${LAMBDA_TASK_ROOT}/src ${LAMBDA_TASK_ROOT}/pyproject.toml ${LAMBDA_TASK_ROOT}/README.md

CMD ["imgeda.lambda_handler.handler.handler"]
