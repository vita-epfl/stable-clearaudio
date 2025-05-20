FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.11-py3

ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID

RUN groupadd --gid ${LDAP_GID} ${LDAP_GROUPNAME} \
    && useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# Install system dependencies and pip
RUN apt update && apt-get install -y \
    python3-pip \
    python3-dev \
    sox \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Copy the source code into the container
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install .

# Install FAD without its dependencies to avoid conflicts
RUN pip install frechet-audio-distance==0.3.1 --no-deps
RUN pip install resampy==0.4.3

# Define working directory
WORKDIR /home/${LDAP_USERNAME}

# Set the final user
USER ${LDAP_USERNAME}
