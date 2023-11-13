# Produce un'immagine Docker di CDK.
# Rif. https://github.com/cdk-team/CDK

FROM alpine

RUN apk add --no-cache wget && \
    wget https://github.com/cdk-team/CDK/releases/download/v1.5.2/cdk_linux_amd64 && \
    chmod a+x cdk_linux_amd64 && \
    mv cdk_linux_amd64 /usr/local/bin/cdk