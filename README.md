# release-deployment-dashboard

Repository for Release Deployment Dashboard Flask App.  
App is deployed on AWS ECS/Fargate and uses Redis caching plus a background scheduler.

---

## What this app is

- **Backend**: Flask (Python) served via **Gunicorn**
- **Caching**: Redis (supports TLS via `REDIS_SSL`)
- **Auth**: GitLab OAuth2 (Group App) + PAT for API access
- **Scheduler**:
  - Local/dev: `startup.sh` runs scheduler + gunicorn together
  - ECS/Fargate: scheduler runs as a **separate container** in the same task definition

The HTML dashboards are served from templates and rely on dynamic project data and prioritization config.

Key repo config files:
- `priority_project_ids.yml`: prioritized GitLab project IDs (gitflow/trunk + categories) used by the UI ordering logic.
- `release-deployment-dashboard-dev.json`: ECS task definition template (includes 2 app containers + Datadog + Firelens) with `${version}` placeholder for the image tag.
- `Dockerfile`: builds a runnable container image.

### App Features
* Flask + Gunicorn architecture
* Scheduler for auto-refreshing project data
* Redis caching for performance
* Security
  * Twingate VPN Access only (when accessing https://release-deployment-dashboard.myorg.com/)
  * OAuth2
  * Session Handling & Management
* Custom Logging
* ECS Fargate for container orchestration

### TechStack
* Backend:
  * Language  :  Python 3.x
  * Framework :  Flask
* Frontend:
  * HTML
  * CSS
  * JavaScript
  * Jinja2
* Web Server
  * Gunicorn (multi-worker, multi-threaded)
* Caching
  * Redis
* Scheduler
  * APScheduler (within Python) or separate container
* Oauth
  * Flask-OAuthlib → GitLab’s OAuth2 ([python-gitlab](https://python-gitlab.readthedocs.io/en/stable/api-usage.html), [GitLab OAuth2](https://docs.gitlab.com/ee/api/oauth2.html#access-gitlab-api-with-access-token))
* HTTP Client
  * requests + requests.adapters + urllib3.util.retry
* Logging
  * Python’s logging module with custom formatters
* CI/CD
  * GitLab CI
* Containerization
  * Docker
* Deployment
  * AWS ECS (Fargate)
* Version Control
  * GitLab

---

## Required environment variables

### Always required
- `GITLAB_ACCESS_TOKEN`
- `GITLAB_APPLICATION_ID`
- `GITLAB_APPLICATION_SECRET`

### Optional (but recommended in real deployments)
- Redis:
  - `REDIS_HOST`
  - `REDIS_PORT` (default 6379)
  - `REDIS_DB` (default 0)
  - `REDIS_SSL` (`True` enables `rediss://`)
- `SESSION_SECRET_KEY` (defaults to a repo-defined fallback)
- `RUN_APP_SCHEDULER`
  - If set to `true`, the app runs in a “production-ish” mode
  - For ECS, scheduler is typically started in a separate container (see below)

---

## Run locally

```bash
export GITLAB_ACCESS_TOKEN="..."
export GITLAB_APPLICATION_ID="..."
export GITLAB_APPLICATION_SECRET="..."
```

Option A — run scheduler and gunicorn separately (to test frequent local HTML changes while the server runs in background):
```bash
# terminal 1
RUN_APP_SCHEDULER=true python scheduler.py
# terminal 2
gunicorn main:app --bind=0.0.0.0:3000 --worker-class=gthread --workers=3 --threads=5 --timeout=120 --log-level=debug
```

Option B — run via helper script for complete hosting with scheduler (kills old processes and starts both):
```bash
./startup.sh
```
This `startup.sh` script ensures old background processes are killed, starts the `scheduler.py` and waits for it to initialize to continuously refresh local Redis cache, and then, starts Gunicorn WSGL server by running `main.py` Flask app.

Now, access the Releases Dashboard at - http://127.0.0.1:3000

---

## Run with Docker

```bash
docker build --compress -t registry.gitlab.com/myorg/release-deployment-dashboard/release-deployment-dashboard:test .

docker run -d -p 3000:3000 \
  -e "GITLAB_ACCESS_TOKEN=$(git config --global gitlab.token)" \
  -e "GITLAB_APPLICATION_ID=<GitLab App ID>" \
  -e "GITLAB_APPLICATION_SECRET=<GitLab App Secret>" 
  registry.gitlab.com/myorg/release-deployment-dashboard/release-deployment-dashboard:test
```

Now, access the Releases Dashboard at - http://127.0.0.1:3000

---

## CI/CD overview: build + deploy to AWS ECS/Fargate

High level:
1. GitLab CI builds a Docker image
2. Image is pushed to Amazon ECR
3. ECS task definition is rendered with the image tag
4. ECS service is updated with a forced new deployment

This repo is deployed using GitLab CI in two phases:

1) Build & Push
   - Pipeline builds a Docker image using Dockerfile, tags it with:
     - CI_COMMIT_TAG (for tag pipelines), OR 
     - CI_COMMIT_SHORT_SHA (for main branch pipelines)
   - Then it pushes to Amazon ECR.

2) Deploy to ECS
   - Pipeline renders the ECS task definition template:
     - release-deployment-dashboard-dev.json
   - It replaces:
     - ${version} → the image tag produced in the build job
   - Then it runs:
     - aws ecs register-task-definition --cli-input-json file://rendered.json
     - aws ecs update-service --force-new-deployment
   - This triggers a new rollout in ECS.

---

## ECS architecture

The task definition uses multiple containers:
- Web container (`release-deployment-dashboard`): Runs the web server via Gunicorn on port 3000 serving Flask app
- Scheduler container (`release-deployment-dashboard-scheduler`): Runs `scheduler.py` to keep caches warm and refresh project data
- Sidecars agents: Datadog + Fluent Bit for logs


This is preferred over running startup.sh inside a single container on ECS because:
  - scheduler lifecycle can be managed independently 
  - web container can scale without duplicating scheduler work

---

## GitLab CI/CD variables

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPO`
- `ECS_CLUSTER`
- `ECS_SERVICE`

Set these in GitLab → Project → Settings → CI/CD → Variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (default: `us-east-1`)
- `ECR_REPO`
  - Example: `<aws-account-id>.dkr.ecr.us-east-1.amazonaws.com/myorg/release-deployment-dashboard`
- `ECS_CLUSTER` (dev cluster name)
- `ECS_SERVICE` (dev service name)
Optional:
- `ECS_TASKDEF_TEMPLATE` (default: `release-deployment-dashboard-dev.json`)

---

### Adding a production deploy

Recommended approach:
1. Copy `release-deployment-dashboard-dev.json` → `release-deployment-dashboard-prod.json`
2. Update prod-only values (cluster/service, Redis endpoints, task roles, etc.)
3. Add GitLab variables for prod, and update `.gitlab-ci.yml` deploy_prod job to use them.
4. Gate prod deploy on tags and keep it `manual`.

### Notes on prioritized projects config

- `priority_project_ids.yml` controls how projects are grouped and ordered in the UI
(e.g: backend/frontend/infrastructure + “others”). Update it as needed when adding/removing projects.
