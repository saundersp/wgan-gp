FROM debian:13.2-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3.13-venv=3.13.5-2 \
	&& ln -sv /usr/bin/python3.13 /usr/bin/python \
	&& apt-get autoremove -y \
	&& apt-get autoclean -y \
	&& rm -rv /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} wgangp \
	&& useradd -m -l -g ${GID} -u ${UID} -G sudo,wgangp wgangp

USER wgangp

WORKDIR /home/wgangp

COPY requirements.txt .
RUN python -m venv --upgrade-deps venv && venv/bin/pip install --no-cache-dir --disable-pip-version-check --timeout 60 --retries 9999999999999999 -r requirements.txt \
	&& rm requirements.txt

COPY *.py .

ENTRYPOINT ["venv/bin/python", "launcher.py"]
