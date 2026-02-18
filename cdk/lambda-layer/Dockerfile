FROM public.ecr.aws/lambda/python:3.12

# Install imgeda and dependencies into the layer structure
COPY . /build/
RUN pip install /build/ -t /opt/python/ && \
    cp -r /build/src/imgeda /opt/python/

# Output the layer contents
CMD ["bash", "-c", "ls -la /opt/python/"]
