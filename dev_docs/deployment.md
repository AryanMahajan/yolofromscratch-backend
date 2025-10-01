# Backend Deployment: Django on Cloud Run via GitHub Actions

This document outlines the modern, serverless deployment strategy for the Django backend.

## 1. High-Level Architecture

-   **Backend (Django + YOLO):** A Docker container running on **Google Cloud Run**, providing a fully-managed, auto-scaling, serverless environment.
-   **Database:** A serverless PostgreSQL database from **Neon DB**.
-   **CI/CD (Backend):** A **GitHub Actions** workflow automates building the Docker image, pushing it to a registry, and deploying to Cloud Run.
-   **Container Registry:** Docker images are stored in **Google Artifact Registry**.
-   **Authentication:** GitHub Actions authenticates with GCP securely using **Workload Identity Federation**, eliminating the need for long-lived service account keys.

---

## 2. Database Setup: Neon DB

1.  **Sign up for Neon:** Create an account.
2.  **Create a Project:** This will contain your database.
3.  **Get Connection String:** From the project dashboard, copy the **Database URL** (connection string). It looks like `postgres://user:password@host:port/dbname`.
4.  **Store Securely:** This URL is a secret. You will add it to your backend repository's **GitHub Secrets**.

---

## 3. Backend Deployment: Cloud Run & GitHub Actions

This pipeline automatically deploys your containerized Django app.

### Step 1: GCP Setup for GitHub Actions

1.  **Enable APIs:** In your GCP project, enable the **Cloud Run API**, **Artifact Registry API**, and **IAM Credentials API**.
2.  **Create Artifact Registry:** Create a **Docker** repository to store your backend images.
3.  **Create a GCP Service Account (SA):** This SA will be used by GitHub Actions.
    -   Grant it the following roles: `Cloud Run Admin`, `Artifact Registry Writer`, `iam.serviceAccountUser`.
4.  **Set up Workload Identity Federation (WIF):** This is the key to secure, keyless authentication.
    -   Go to `IAM & Admin > Workload Identity Federation`.
    -   Create a new **Pool**.
    -   Create a **Provider** within the pool. Select **OpenID Connect (OIDC)**.
        -   **Issuer (URL):** `https://token.actions.githubusercontent.com`
        -   **Audience:** Use the default.
        -   **Attribute Mapping:** Map `google.subject` to `assertion.sub`. The subject (`sub`) from GitHub must be in the format `repo:your-github-org/your-repo-name:ref:refs/heads/main`.
    -   **Connect SA:** After creating the provider, allow the Service Account you created in the previous step to be impersonated by identities from this provider. Grant the SA the "Workload Identity User" role for federated identities that match your repository.

### Step 2: GitHub Repository Configuration

1.  **Add GitHub Secrets:** In your backend's GitHub repository, go to `Settings > Secrets and variables > Actions` and add the following secrets:
    -   `GCP_PROJECT_ID`: Your GCP Project ID.
    -   `GCP_WORKLOAD_IDENTITY_PROVIDER`: The full resource name of the WIF provider you created. Find it in the WIF provider details.
    -   `GCP_SERVICE_ACCOUNT`: The email address of the GCP Service Account.
    -   `NEON_DB_URL`: The connection string from Neon DB.

2.  **Create the GitHub Actions Workflow:**
    -   In your backend repository, create the file `.github/workflows/deploy.yml`.

    ```yaml
    name: Build and Deploy to Cloud Run

    on:
      push:
        branches:
          - main

    env:
      GCP_REGION: us-central1 # Change to your preferred region
      APP_NAME: yolo-backend
      ARTIFACT_REPO: yolo-repo # Your Artifact Registry repo name

    jobs:
      build-and-deploy:
        name: Build and Deploy
        runs-on: ubuntu-latest
        permissions:
          contents: 'read'
          id-token: 'write' # Required for Workload Identity Federation

        steps:
        - name: Checkout
          uses: actions/checkout@v3

        - name: Authenticate to Google Cloud
          id: auth
          uses: google-github-actions/auth@v1
          with:
            workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
            service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

        - name: Set up Cloud SDK
          uses: google-github-actions/setup-gcloud@v1

        - name: Configure Docker
          run: gcloud auth configure-docker ${{ env.GCP_REGION }}-docker.pkg.dev

        - name: Build and Push Docker Image
          run: |
            docker build -t "${{ env.GCP_REGION }}-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/${{ env.ARTIFACT_REPO }}/${{ env.APP_NAME }}:${{ github.sha }}" .
            docker push "${{ env.GCP_REGION }}-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/${{ env.ARTIFACT_REPO }}/${{ env.APP_NAME }}:${{ github.sha }}"

        - name: Deploy to Cloud Run
          id: deploy
          uses: google-github-actions/deploy-cloudrun@v1
          with:
            service: ${{ env.APP_NAME }}
            region: ${{ env.GCP_REGION }}
            image: "${{ env.GCP_REGION }}-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/${{ env.ARTIFACT_REPO }}/${{ env.APP_NAME }}:${{ github.sha }}"
            platform: 'managed'
            allow_unauthenticated: true # Set to false if you want to control access
            env_vars: |
              DATABASE_URL=${{ secrets.NEON_DB_URL }}
              DJANGO_SETTINGS_MODULE=core.settings

        - name: Output Cloud Run URL
          run: echo "Backend deployed to ${{ steps.deploy.outputs.url }}"
    ```

### Step 3: Application `Dockerfile`

The `Dockerfile` in `yolofromscratch` should be configured to use the `$PORT` environment variable provided by Cloud Run.

```Dockerfile
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Gunicorn will listen on the port specified by the PORT env var.
# Cloud Run sets this automatically (typically 8080).
CMD exec gunicorn core.wsgi:application --bind 0.0.0.0:$PORT --workers 2
```