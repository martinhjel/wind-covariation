FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Modifications to add non-root user: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG CONDA_ENV=wind

ARG CONDA_INSTALL_PATH=/opt/conda

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo openssh-client \    
    && DEBIAN_FRONTEND=noninteractive \    
    && apt-get dist-upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends apt-utils \    
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \    
    && chown -R ${USERNAME} ${CONDA_INSTALL_PATH} 

# # [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Install conda environment packages and initialize conda in bash shell
COPY environment.yml .
RUN conda init bash \
    && conda env update --file environment.yml 

# Make RUN commands use the new environment:
RUN echo "conda activate wind" >> ~/.bashrc
ENV PATH /opt/conda/envs/wind/bin:$PATH