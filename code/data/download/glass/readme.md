# 🌍 GLASS File Downloader and GCS Uploader

This project downloads geospatial files from the [GLASS archive](https://glass.hku.hk/) and uploads them to a specified Google Cloud Storage (GCS) bucket. The containerized app is designed to run once (e.g. via a Kubernetes Job), checking what files already exist in GCS, downloading missing files, and uploading them.

---

## 🧱 Project Structure

```text
.
├── download/           # Logic for file discovery and download
├── gcs/                # GCS upload logic
├── workflow.py         # Orchestrates download/upload
├── config.py           # Configuration using env vars or defaults
├── main.py             # Entry point
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── glass-job.yaml      # Kubernetes Job definition
└── README.md
```

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
docker build -t gcr.io/your-project-id/glass-uploader:latest .
docker push gcr.io/your-project-id/glass-uploader:latest
```

---

## ☁️ Run Once on Google Kubernetes Engine (GKE)

### 🧱 Step 1: Create GKE Cluster (if not already done)

```bash
gcloud container clusters create-auto glass-cluster \
  --region=us-central1
```

### 🔐 Step 2: Enable Workload Identity on Cluster

```bash
gcloud container clusters update glass-cluster \
  --region=us-central1 \
  --workload-pool=your-project-id.svc.id.goog
```

### 👤 Step 3: Create Google Service Account (GSA)

```bash
gcloud iam service-accounts create glass-uploader-sa \
  --display-name="Glass Uploader"
```

Grant it access to write to GCS:

```bash
gcloud projects add-iam-policy-binding your-project-id \
  --member="serviceAccount:glass-uploader-sa@your-project-id.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

### 🤝 Step 4: Create Kubernetes Service Account (KSA) and Bind It

```bash
kubectl create serviceaccount k8s-glass-uploader

gcloud iam service-accounts add-iam-policy-binding glass-uploader-sa@your-project-id.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:your-project-id.svc.id.goog[default/k8s-glass-uploader]"

kubectl annotate serviceaccount \
  k8s-glass-uploader \
  iam.gke.io/gcp-service-account=glass-uploader-sa@your-project-id.iam.gserviceaccount.com
```

---

### 📄 Step 5: Deploy the Kubernetes Job

Edit `glass-job.yaml` and replace:

- `your-project-id`
- `your-bucket-name`

Then apply:

```bash
kubectl apply -f glass-job.yaml
```

Check logs:

```bash
kubectl logs -l job-name=glass-uploader-job
```

---

## ✅ Clean Up (Optional)

```bash
kubectl delete job glass-uploader-job
kubectl delete serviceaccount k8s-glass-uploader
gcloud iam service-accounts delete glass-uploader-sa@your-project-id.iam.gserviceaccount.com
```

---

## 📦 Example GCS Upload Path

Files from:

```
https://glass.hku.hk/archive/LST/MODIS/Daily/1KM/2021/file1.hdf
```

Are uploaded to:

```
gs://your-bucket-name/GCS_PATH_PREFIX/LST/MODIS/Daily/1KM/file1.hdf
```

Prefix and data type are automatically extracted.

---

## 📄 License

MIT License
