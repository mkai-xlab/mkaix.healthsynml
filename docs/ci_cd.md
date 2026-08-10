# CI and Deployment

## CI

GitHub Actions runs tests for pushes and pull requests to `dev` and `main`.

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q --disable-warnings --maxfail=1
```

CI does not store or download checkpoints. Tests that need real weights skip when checkpoints are unavailable.

## Deployment

- Pushes to `dev` deploy to the development EC2 host.
- Production deployment is manual from the `main` branch.

Both hosts need Docker Compose, the project checkout, `../env/ai.env`, the external `knee-oa-net` network, and the read-only `checkpoints/` directory.

Required repository secrets:

| Environment | Secrets |
| --- | --- |
| Development | `EC2_PUBLIC_IP`, `SSH_PRIVATE_KEY` |
| Production | `VIETTEL_PUBLIC_IP`, `VIETTEL_PASSWORD` |

After deployment, verify the service:

```bash
curl --fail http://127.0.0.1:8005/api/v1/health
```

To roll back, deploy the last reviewed commit and rebuild the `ai` service. Never put checkpoints or host configuration in GitHub secrets or repository files.
