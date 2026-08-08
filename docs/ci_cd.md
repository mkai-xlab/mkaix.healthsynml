# CI/CD Pipeline

This project uses GitHub Actions for unit-test validation and controlled
container delivery. Checkpoints are not stored in Git, so CI validates code and
unit tests only; model-loading smoke tests run only where read-only checkpoints
are mounted.

## Continuous Integration

Workflow: `.github/workflows/build-and-test.yml`

CI runs for pull requests to `dev` or `main`, and for direct pushes to those
branches when application, test, dependency, Docker, or workflow files change.
It uses Python 3.10 and installs `requirements-ci.txt`. That file pins the test
environment and obtains CPU-only `torch` and `torchvision` wheels, preventing
GitHub-hosted runners from downloading unused CUDA libraries.

The unit-test command is:

```bash
python -m pytest -q --disable-warnings --maxfail=1
```

Run the equivalent test suite locally after installing the CI dependencies:

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q --disable-warnings --maxfail=1
```

Tests that require a real checkpoint skip when it is not mounted. This is
intentional: checkpoints are ignored by Git and must never be committed to make
CI pass.

## Continuous Delivery

`aws-ec2-deploy.yml` deploys commits pushed to `dev` to the development EC2
host. It connects over SSH, fast-forwards `dev`, validates the Compose file,
rebuilds the `ai` service, and removes unused Docker images.

`viettel-idc-prod-deploy.yml` is a manual production release. In the Actions
page, choose **Deliver ML Service to Viettel IDC**, select the latest `main`
revision, and set `confirm_delivery` to true. The workflow then fast-forwards
`main` and rebuilds the same service on the production host.

Required repository secrets:

| Workflow | Secrets |
| --- | --- |
| AWS development | `EC2_PUBLIC_IP`, `SSH_PRIVATE_KEY` |
| Viettel IDC production | `VIETTEL_PUBLIC_IP`, `VIETTEL_PASSWORD` |

Each target host must already contain the repository checkout, Docker Compose,
the external `knee-oa-net` network, `../env/ai.env`, and the read-only
`checkpoints/` directory. Do not put these values or model files in GitHub
Actions secrets, logs, or repository files.

## Release Checks And Rollback

After deployment, verify readiness from the target host or its approved network:

```bash
curl --fail http://127.0.0.1:8005/api/v1/health
```

For a rollback, identify the last known-good commit, fast-forward or reset the
deployment checkout to that reviewed commit according to the environment's
change policy, then run:

```bash
docker compose up -d --build ai
curl --fail http://127.0.0.1:8005/api/v1/health
```

Production releases remain manual so a passing unit-test job cannot deploy a
clinical inference service without an explicit operator confirmation.
