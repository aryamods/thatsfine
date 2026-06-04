FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at BUILD TIME (not runtime) so Railway never re-downloads it
# This bakes the model into the image — no quota issues, instant cold start
RUN python -c "\
import gdown, h5py, os, sys; \
url = 'https://drive.google.com/uc?id=1Tm1OpVsGDvGtCHNaq4xu_DVNzjCSZO3X'; \
gdown.download(url, 'skin_model.h5', quiet=False, fuzzy=True, use_cookies=False); \
size = os.path.getsize('skin_model.h5') / 1024**2; \
print(f'Downloaded: {size:.1f} MB'); \
h5py.File('skin_model.h5', 'r').close(); \
print('HDF5 signature OK') \
"

COPY . .

CMD ["python", "app.py"]
