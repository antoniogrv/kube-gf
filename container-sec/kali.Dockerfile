# Produce un'immagine Docker di Kali Linux per le analisi di Container Security e l'esecuzione di attacchi di Penetration Testing.
# Fare riferimento alla documentazione del progetto per maggiori dettagli.

FROM kalilinux/kali-rolling@sha256:b6bd6a9e6f62171e4e0d8f43f4b0d02f1df0ba493225da852968576bc2d602d2

WORKDIR /root
ENV DEBIAN_FRONTEND=noninteractive

# L'immagine di Kali Linux non include di default il pacchetto kali-linux-headless ..
# .. che contiene la maggior parte degli strumenti di analisi e attacco.
# Rif. https://www.kali.org/docs/containers/using-kali-docker-images/

RUN apt -y update && \
    apt -y dist-upgrade && \
    apt -y autoremove && \
    apt clean && \
    apt -y install wget

# Siamo interessati esclusivamente a Metasploit Framework, contenuto nel pacchetto kali-linux-top10.
# Immagini più organiche potrebbero utilizzare il pacchetto kali-linux-all.
# Rif. https://www.kali.org/blog/kali-linux-metapackages/

RUN apt -y install -f kali-tools-top10

# Iniezione arbitraria del binario del Zero Dependency Container Penetration Toolkit
# Rif. https://github.com/cdk-team/CDK

RUN wget https://github.com/cdk-team/CDK/releases/download/v1.5.2/cdk_linux_amd64 && \
    chmod a+x cdk_linux_amd64 && \
    mv cdk_linux_amd64 /usr/local/bin/cdk