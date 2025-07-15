# 🌍 File Downloader and GCS Uploader

This project downloads data files from one or more **pluggable sources** (e.g. [GLASS archive](https://glass.hku.hk/), [ESA CCI](https://climate.esa.int/en/projects/climate-data-store/)) and uploads them to a specified Google Cloud Storage (GCS) bucket.

The containerized app is designed to run once (e.g. via a Kubernetes Job), checking which files already exist in GCS, downloading missing files, and uploading them. It supports multiple data sources using a unified interface (`BaseDataSource`), making it easy to extend.

---

## 🧱 Project Structure

```text
.
├── download/           # Logic for file discovery and data source classes
│   ├── base.py         # Abstract base class for data sources
│   ├── glass.py        # GLASS-specific implementation
│   ├── esacci.py       # ESA CCI-specific implementation
│   └── ...
├── gcs/                # GCS upload logic
├── tests/              # Unit tests
├── workflow.py         # Generalized download/upload workflow
├── config.py           # Configuration using environment variables
├── main.py             # Entry point
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── glass-job.yaml      # Kubernetes Job definition
└── README.md
```

---

## ⚙️ Configuration via Environment Variables

These values can be set via your Kubernetes job (see below), `.env`, or your deployment system:

| Variable         | Description                                             | Example Value                                                   |
|------------------|---------------------------------------------------------|------------------------------------------------------------------|
| `GCP_PROJECT_ID` | Your Google Cloud project ID                            | `ee-growthandheat`                                              |
| `GCS_BUCKET_NAME`| Name of the GCS bucket                                  | `growthandheat`                                                 |
| `DATA_SOURCE_NAME`| Identifier for which data source to use                | `glass`, `esacci`                                               |
| `BASE_URL`       | URL to crawl/download files from                        | `https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/`             |

---

## 🐳 Build & Push the Container to GCR

### 1. **Authenticate with Google Cloud**

```bash
gcloud auth login
gcloud config set project your-project-id
```

### 2. **Enable Required Services**

```bash
gcloud services enable containerregistry.googleapis.com
```

### 3. **Build and Push**

```bash
docker build -t gcr.io/your-project-id/uploader:latest .
docker push gcr.io/your-project-id/uploader:latest
```

---

## ☁️ Run Once on Google Kubernetes Engine (GKE)

### 🧱 Step 1: Create GKE Cluster (if not already done)

```bash
gcloud container clusters create-auto uploader-cluster \
  --region=us-central1
```

### 🔐 Step 2: Enable Workload Identity on Cluster

```bash
gcloud container clusters update uploader-cluster \
  --region=us-central1 \
  --workload-pool=your-project-id.svc.id.goog
```

### 👤 Step 3: Create Google Service Account (GSA)

```bash
gcloud iam service-accounts create uploader-sa \
  --display-name="Data Uploader"
```

Grant it access to write to GCS:

```bash
gcloud storage buckets add-iam-policy-binding gs://your-bucket-name --member="serviceAccount:uploader-sa@your-project-id.iam.gserviceaccount.com" --role="roles/storage.objectAdmin"
```

### 🤝 Step 4: Create Kubernetes Service Account (KSA) and Bind It

```bash
kubectl create serviceaccount k8s-uploader

gcloud iam service-accounts add-iam-policy-binding uploader-sa@your-project-id.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:your-project-id.svc.id.goog[default/k8s-uploader]"

kubectl annotate serviceaccount \
  k8s-uploader \
  iam.gke.io/gcp-service-account=uploader-sa@your-project-id.iam.gserviceaccount.com
```

---

### 📄 Step 5: Deploy the Kubernetes Job

Optionally, provide eog credentials with

```bash
kubectl create secret generic eog-login \
  --from-literal=EOG_USERNAME='your_username' \
  --from-literal=EOG_PASSWORD='your_password'
```

Update `glass-job.yaml`:

- Replace `your-project-id`
- Replace `your-bucket-name`
- Add or override `env` variables as needed (e.g. `BASE_URL`, `DATA_SOURCE_NAME`)

Then apply:

```bash
kubectl apply -f glass-job.yaml
```

Check logs:

```bash
kubectl logs -l job-name=glass-uploader-job
```

---

## 🧪 Adding New Data Sources

To support a new data source:

1. Create a new file in `download/` (e.g., `chirps.py`)
2. Inherit from `BaseDataSource`
3. Implement:
   - `list_remote_files()`
   - `download(...)`
   - `local_path(...)`
   - `gcs_upload_path(...)`
4. Register it in `config.py`:

```python
from download.chirps import CHIRPSDataSource
from download.esacci import ESACCIDataSource

DATA_SOURCES = {
    "glass": GLASSDataSource,
    "esacci": ESACCIDataSource,
    "chirps": CHIRPSDataSource,
}
```

5. Set `DATA_SOURCE_NAME=chirps` in your Kubernetes Job.

---

## 📦 Example GCS Upload Path

A file from:

```
https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/2021/file1.hdf
```

Will be uploaded to:

```
gs://your-bucket-name/GCS_PATH_PREFIX/MODIS/Daily/1KM/file1.hdf
```

The prefix (`GCS_PATH_PREFIX`) and remote path are parsed automatically based on the base URL and file structure.

---

## 📄 License

MIT License