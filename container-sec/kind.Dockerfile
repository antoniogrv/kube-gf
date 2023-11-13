# Produce un'immagine Docker di un worker node di Kind personalizzata per ospitare un'istanza di CDK.

FROM kindest/node:v1.27.3

# Iniezione arbitraria del binario del Zero Dependency Container Penetration Toolkit
# Rif. https://github.com/cdk-team/CDK

RUN apt update && \
    apt -y install wget && \
    wget https://github.com/cdk-team/CDK/releases/download/v1.5.2/cdk_linux_amd64 && \
    chmod a+x cdk_linux_amd64 && \
    mv cdk_linux_amd64 /usr/local/bin/cdk