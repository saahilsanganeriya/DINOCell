# AWS S3 Streaming for JUMP Dataset

Pretraining can read JUMP Cell Painting images straight from the public `cellpainting-gallery` S3 bucket, so the dataset does not have to be downloaded first.

## Benefits

- **No full download**: images are fetched when the data loader asks for them
- **LRU cache**: the 1000 most recently used images stay in RAM
- **No credentials**: the bucket is public and is read with unsigned requests
- **Only what is read**: images the run never touches are never fetched

## Quick Start

### Prerequisites

```bash
# Install S3 dependencies
pip install boto3 smart-open
```

### Test S3 Access

```bash
python -c "
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Create S3 client (no credentials needed!)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# List some objects
response = s3.list_objects_v2(
    Bucket='cellpainting-gallery',
    Prefix='cpg0000-jump-pilot/source_4/images/',
    MaxKeys=10
)

print(f'✅ S3 works! Found {len(response.get(\"Contents\", []))} files')
for obj in response.get('Contents', [])[:3]:
    print(f\"  - {obj['Key']}\")
"
```

If this lists files, streaming will work.

## Usage

### Option 1: Launch Script (Recommended)

```bash
cd training/ssl_pretraining
./launch_ssl_with_s3_wandb.sh
```

The script automatically:
- Detects S3 streaming from config
- Tests S3 connectivity
- Launches training

### Option 2: Manual Configuration

Edit config file: `configs/dinov3_vits8_jump_s3_multiview.yaml`

```yaml
train:
  dataset_path: 'JUMPS3MultiView:bucket=cellpainting-gallery:prefix=cpg0000-jump-pilot/source_4/images'
```

Then run, from the repository root:
```bash
cd dinov3_modified/dinov3
PYTHONPATH=. python dinov3/train/train.py \
    --config-file ../../training/ssl_pretraining/configs/dinov3_vits8_jump_s3_multiview.yaml \
    --output-dir ../../training/ssl_pretraining/output
```

## How It Works

### Architecture

```
Training Loop
  ↓
DataLoader requests batch
  ↓
S3Dataset.__getitem__(index)
  ↓
Check LRU cache
  ├─ Hit → Return cached image
  └─ Miss → Download from S3
       ↓
     boto3.client.get_object()
       ↓
     Decode TIFF
       ↓
     Apply CLAHE preprocessing
       ↓
     Add to LRU cache (evict old if full)
       ↓
     Return image
```

### Cache Strategy

- **Size**: 1000 images
- **Policy**: LRU (least recently used)

Hit rate and bandwidth were not measured.

### S3 Bucket Structure

```
s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/
├── 2020_11_04_CPJUMP1/
│   └── images/
│       ├── BR00117010__2020-11-05T19_51_18-Measurement1/
│       │   └── Images/
│       │       ├── r01c01f01p01-ch1sk1fk1fl1.tiff  # Channel 1
│       │       ├── r01c01f01p01-ch2sk1fk1fl1.tiff  # Channel 2
│       │       ├── r01c01f01p01-ch3sk1fk1fl1.tiff  # Channel 3
│       │       ├── r01c01f01p01-ch4sk1fk1fl1.tiff  # Channel 4
│       │       ├── r01c01f01p01-ch5sk1fk1fl1.tiff  # Channel 5
│       │       └── ... (brightfield channels ch6-8)
│       └── ... (more plates)
└── ... (more batches)
```


## Advanced Configuration

### Adjust Cache Size

In `dinov3_modified/dinov3/data/datasets/jump_cellpainting_s3.py`:

```python
class JUMPS3Dataset(ExtendedVisionDataset):
    def __init__(
        self,
        ...
        cache_size: int = 1000,  # Change this!
    ):
```

The cache lives in host RAM, not on the GPU, so size it to the machine's memory.

### Use Different S3 Bucket

If you have JUMP data in your own S3:

```yaml
train:
  dataset_path: 'JUMPS3MultiView:bucket=my-bucket:prefix=my-jump-data'
```

### Fallback to Local

If S3 fails, use local dataset:

```yaml
train:
  dataset_path: 'JUMPCellPaintingMultiView:root=/path/to/local/jump'
```

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'boto3'`

```bash
pip install boto3 smart-open
```

### Error: `S3 connection timeout`

Check internet connection:
```bash
ping cellpainting-gallery.s3.amazonaws.com
```

### Error: `Access Denied`

The bucket is public, but double-check:
```bash
aws s3 ls s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/ --no-sign-request
```

### Slow download speeds

- Check your bandwidth: `speedtest-cli`
- Increase cache size if you have RAM
- Consider downloading a subset locally

## Streaming or local files

Streaming trades disk space for network reads: nothing is downloaded up front, but every cache miss is an S3 request. Local files avoid the network once downloaded. The two were not timed against each other in this repository.

## How S3 Streaming Is Implemented

### 1. Custom Dataset Class

`dinov3_modified/dinov3/data/datasets/jump_cellpainting_s3.py`:

```python
class JUMPS3Dataset(ExtendedVisionDataset):
    def __init__(self, bucket, prefix, ...):
        # Create unsigned S3 client (no auth)
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        
        # Discover all images without downloading
        self._discover_s3_images(batches, max_samples)
    
    def __getitem__(self, index):
        # Load from cache or S3
        s3_paths = self.fields[index]
        channels = [self._load_from_s3(path) for path in s3_paths]
        return preprocess(channels)
```

### 2. LRU Cache

A dictionary plus an access-order list:
```python
class S3ImageCache:
    def __init__(self, cache_size=1000):
        self._cache = {}
        self._access_order = []
    
    def get(self, key):
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        if len(self._cache) >= self.cache_size:
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = value
        self._access_order.append(key)
```

### 3. Integration with DINOv3

Registered in `dinov3_modified/dinov3/data/loaders.py`:

```python
def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")
    name = tokens[0]
    
    if name == "JUMPS3MultiView":
        class_ = JUMPS3Dataset
        kwargs['channel_mode'] = 'multiview'
        kwargs['bucket'] = 'cellpainting-gallery'
        kwargs['prefix'] = 'cpg0000-jump-pilot/source_4/images'
    ...
```

## Public Dataset Access

The JUMP Cell Painting dataset is publicly available:
- **Website**: https://jump-cellpainting.broadinstitute.org/
- **S3 Browser**: https://registry.opendata.aws/cellpainting-gallery/
- **Paper**: https://www.nature.com/articles/s41592-024-02241-6

No registration or credentials are needed.


