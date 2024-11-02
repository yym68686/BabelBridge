# BabelBridge

Docker build

```bash
docker build --no-cache -t babelbridge:latest -f Dockerfile --platform linux/amd64 .
docker tag babelbridge:latest yym68686/babelbridge:latest
docker push yym68686/babelbridge:latest
```