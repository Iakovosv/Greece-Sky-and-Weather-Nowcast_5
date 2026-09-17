FROM ghcr.io/home-assistant/aarch64-base:latest
ENV LANG=C.UTF-8
RUN apk add --no-cache python3 py3-requests py3-paho-mqtt py3-numpy
COPY rootfs /
RUN chmod +x /run.sh
CMD ["/run.sh"]
