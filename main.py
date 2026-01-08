#!/usr/bin/env python

########################################################################################################################
####################### Flask App to dynamically renders HTML templates based on user selection ########################
########################################################################################################################
##
## REQUIRED SYSTEM LEVEL variables to run the script:
##   GITLAB_ACCESS_TOKEN            :   Used for accessing GitLab APIs during initial Server Start
##   GITLAB_APPLICATION_ID          :   GitLab's Group Application ID
##   GITLAB_APPLICATION_SECRET      :   GitLab's Group Application Secret
##   [OPTIONAL] Redis configuration :   REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_SSL
##   [OPTIONAL] SESSION_SECRET_KEY  :   Any secret to secure browser cookies; Default: 'th3_b35t_v1dm0b_s3cr3t'
##   [OPTIONAL] RUN_APP_SCHEDULER   :   If set, App server will be configured with scheduled restarts
########################################################################################################################

from flask import Flask, request, render_template, jsonify, redirect, url_for, session, make_response, has_request_context
from flask_oauthlib.client import OAuth
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import atexit
import logging
import threading
import gitlab
from gitlab import GitlabAuthenticationError
# Constants for user access level values: GUEST_ACCESS(10), REPORTER_ACCESS(20), DEVELOPER_ACCESS(30), MAINTAINER_ACCESS(40), OWNER_ACCESS(50)
from gitlab import const
import os
import sys
import re
import yaml
from datetime import datetime, timedelta
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_caching import Cache
import redis
from functools import wraps
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InvalidChunkLength
from dateutil.parser import parse
import random
import json


redis_cache_timeout_project_id = 0          # persistent until refreshed
redis_cache_timeout_project_data = 1800     # non-persistent for 30 min
redis_cache_timeout_project_metrics = 1800  # non-persistent for 30 min
redis_cache_timeout_default = 900           # non-persistent for 15 min

flask_thread_pool_executor_max_workers = 30 # number of parallel processing requests

cache_refresh_lock = threading.Lock()       # In-process threading lock to prevent multiple cache refreshes at the same time

# Flask scheduler time configuration
now = datetime.now().replace(second=0, microsecond=0)
minutes = ((now.minute // 10) + 1) * 10
next_run = now.replace(minute=minutes % 60) + timedelta(hours=(minutes // 60))
# Add 8 minutes to the next_run value to wait until the initial server refreshes (metrics ~5min, data ~3min)
next_run_with_offset = next_run + timedelta(minutes=10)


app = Flask(__name__)
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)
## Optional environment variables: REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_SSL
## SSL Connection must be enabled for Redis hosted on AWS
cache = Cache(app, config={
    'CACHE_TYPE': 'RedisCache',
    'CACHE_REDIS_URL': (
        f"rediss://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}"
        if os.getenv('REDIS_SSL')
        else f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/{os.getenv('REDIS_DB', '0')}"
    ),
    'CACHE_DEFAULT_TIMEOUT': redis_cache_timeout_default,  # 15 min
})
cache.init_app(app)

# Secure Flask Configuration
app.config.update(
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=15),   # Set session timeout to 15 minutes
    SECRET_KEY=os.getenv('SESSION_SECRET_KEY', 'th3_b35t_v1dm0b_s3cr3t'),   # Random secret for session cookies security
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)

gitlab_base_url = 'https://gitlab.com'
server_gitlab_access_token = os.getenv('GITLAB_ACCESS_TOKEN') # GitLab token for getting project IDs at server start
oauth2_client_id = os.getenv('GITLAB_APPLICATION_ID')  # GitLab Application ID
oauth2_client_secret = os.getenv('GITLAB_APPLICATION_SECRET')  # GitLab Application Secret
server_user_name = 'System User'

if not server_gitlab_access_token:
    print("Missing required system level environment variable - GITLAB_ACCESS_TOKEN")
    sys.exit(1)

if not oauth2_client_id:
    print("Missing required system level environment variable - GITLAB_APPLICATION_ID")
    sys.exit(1)

if not oauth2_client_secret:
    print("Missing required system level environment variable - GITLAB_APPLICATION_SECRET")
    sys.exit(1)

#################### Set up custom logging ########################
class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.user_name = server_user_name
        return super().format(record)

class RequestHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            record.message = record.getMessage()
            self.format(record)
            self.stream.write(self.format(record) + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class SingleLineHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            record.message = record.getMessage()
            self.format(record)
            self.stream.write(self.format(record))
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class ShortThreadNameFilter(logging.Filter):
    def filter(self, record):
        name = threading.current_thread().name
        short = name.replace("ThreadPoolExecutor-", "")
        if "_" in short:
            prefix, suffix = short.split("_", 1)
            try:
                suffix_int = int(suffix)
                short = f"{prefix}_{suffix_int:02d}"
            except ValueError:
                pass  # fallback to raw if not an integer
        record.shortThreadName = short
        return True

###### Custom format logger for server logs #####
server_log_handler = RequestHandler()
server_log_handler.addFilter(ShortThreadNameFilter())
server_log_formatter = RequestFormatter('%(levelname)s: [%(asctime)s] [PID:%(process)d THREAD:%(shortThreadName)s] [%(user_name)s] %(message)s')
server_log_handler.setFormatter(server_log_formatter)

server_logger = logging.getLogger('background')
server_logger.setLevel(logging.INFO)
server_logger.addHandler(server_log_handler)
#################################################

###### Custom format logger for app tasks #####
app_log_handler = RequestHandler()
app_log_handler.addFilter(ShortThreadNameFilter())
app_log_formatter = RequestFormatter('%(message)s')
app_log_handler.setFormatter(app_log_formatter)

# Get the 'werkzeug' logger used by Flask
app_logger = logging.getLogger('werkzeug')
app_logger.setLevel(logging.INFO)
app_logger.addHandler(app_log_handler)
#################################################

###### Custom format logger for single line #####
single_line_handler = SingleLineHandler()
single_line_handler.addFilter(ShortThreadNameFilter())
single_line_formatter = RequestFormatter('%(levelname)s: [%(asctime)s] [PID:%(process)d THREAD:%(shortThreadName)s] %(message)s')
single_line_handler.setFormatter(single_line_formatter)

single_line_logger = logging.getLogger(__name__)
single_line_logger.setLevel(logging.INFO)
single_line_logger.addHandler(single_line_handler)
#################################################

def log_request_context(message):
    if has_request_context() and 'gitlab_user_name' in session:
        single_line_logger.info(f"[{session['gitlab_user_name']}] [{message}]")
    else:
        single_line_logger.info(f"[{server_user_name}] [{message}] \n")

def log_error_context(message):
    if has_request_context() and 'gitlab_user_name' in session:
        single_line_logger.error(f"[{session['gitlab_user_name']}] [{message}]")
    else:
        single_line_logger.error(f"[{server_user_name}] [{message}] \n")

##################################################################
## GitLab Authentication
########################
## OAuth2 configuration
oauth = OAuth(app)
gitlab_auth = oauth.remote_app(
    'gitlab',
    consumer_key=oauth2_client_id,
    consumer_secret=oauth2_client_secret,
    request_token_params={'scope': 'api', 'prompt': 'login'},
    base_url=f'{gitlab_base_url}/api/v4/',
    access_token_url=f'{gitlab_base_url}/oauth/token',
    authorize_url=f'{gitlab_base_url}/oauth/authorize',
    request_token_url=None,
    access_token_method='POST'
)

def get_resilient_gitlab_session(retries=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,  # Retry up to n times,
        # read=retries,
        # connect=retries,
        backoff_factor=backoff_factor,  # Wait 1s, 2s, 4s between retries
        status_forcelist=status_forcelist,  # Retry on these
        raise_on_status=False,
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def get_gitlab_client(token):
    resilient_session = get_resilient_gitlab_session()
    return gitlab.Gitlab(
        url=gitlab_base_url,
        oauth_token=token,
        timeout=30,
        session=resilient_session
    )

def fetch_with_retries(callable_func, max_retries=3, backoff=0.5):
    for attempt in range(max_retries):
        try:
            return callable_func()
        except Exception as e:
            # Check for InvalidChunkLength (by type or message)
            if isinstance(e, InvalidChunkLength) or "InvalidChunkLength" in str(e):
                if attempt < max_retries - 1:
                    sleep_time = backoff * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_time)
                    continue
            raise  # Reraise for other errors OR after max retries

##################################################################
def fetch_api_data():
    new_gitflow, new_trunk, non_cicd = [], [], []

    try:
        gl = get_gitlab_client(server_gitlab_access_token)
        gl.auth()
        global server_user_name
        server_user_name = gl.user.name
        projects = fetch_with_retries(lambda: gl.projects.list(get_all=True, iterator=True, visibility='private', archived=False, order_by='last_activity_at', sort='desc'))

        print("******************************************************************")
        print(f"Total Projects: {len(projects)}")
        print("******************************************************************")
        server_logger.info("Total Projects - {}".format(len(projects)))
        counter = 0

        for project in projects:
            try:
                if not project.jobs_enabled or not project.name.startswith("myorg-"):
                    non_cicd.append(project.id)
                    counter += 1
                    server_logger.info("[{}] Non CI/CD - {} ({})".format(counter, project.name, project.id))
                    continue
                envs = project.environments.list(get_all=True)
                if len(envs) < 3:
                    non_cicd.append(project.id)
                    counter += 1
                    server_logger.info("[{}] Non CI/CD - {} ({})".format(counter, project.name, project.id))
                    continue
                if project.default_branch == 'develop':
                    new_gitflow.append(project.id)
                    counter += 1
                    server_logger.info("[{}] GitFlow - {} ({})".format(counter, project.name, project.id))
                else:
                    new_trunk.append(project.id)
                    counter += 1
                    server_logger.info("[{}] Trunk - {} ({})".format(counter, project.name, project.id))
            except Exception as e:
                non_cicd.append(project.id)
                counter += 1
                server_logger.error(f"Error processing project {project.name}: {e}")
    except Exception as e:
        server_logger.error(f"Error fetching projects: {e}")

    print("****************************************")
    server_logger.info("GitFlow Projects - {}".format(len(new_gitflow)))
    server_logger.info("Trunk Projects - {}".format(len(new_trunk)))
    server_logger.info("Non CI/CD Projects - {}".format(len(non_cicd)))
    print("****************************************")
    print(f"GitFlow Projects ({len(new_gitflow)}): {new_gitflow}")
    print("****************************************")
    print(f"Trunk Projects ({len(new_trunk)}): {new_trunk}")
    print("****************************************")
    print(f"Non CI/CD Projects ({len(non_cicd)}): {non_cicd}")
    print("****************************************")

    return new_gitflow, new_trunk

def load_priority_project_ids():
    """Read priority_ids.yml exactly once at startup."""
    priority_projects_file = os.path.join(os.path.dirname(__file__), 'priority_project_ids.yml')
    try:
        with open(priority_projects_file, 'r') as f:
            data = yaml.safe_load(f)
            return data or {'gitflow': [], 'trunk': []}
    except FileNotFoundError:
        app.logger.warning(f"{priority_projects_file} not found; proceeding without priorities...")
        return {'gitflow': [], 'trunk': []}

def verify_redis_cache():
    try:
        r = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            ssl=(os.getenv('REDIS_SSL', 'True').lower() in ['true', '1', 'yes'])
        )
        r.ping()
        print("Redis is reachable ✅")
    except redis.ConnectionError as e:
        print(f"Redis connection failed ❌: {e}")
        #sys.exit(1)

def get_project_updated_at_cache_key(workflow, project_id):
    return f"project_data_{workflow}_{project_id}_updated_at"

def get_project_data_cache_key(workflow, project_id):
    return f"project_data_{workflow}_{project_id}"

def get_project_metrics_cache_key(project_id):
    return f"project_metrics_{project_id}"

def set_project_id_cache():
    gitflow, trunk = fetch_api_data()
    try:
        cache.set('gitflow_project_ids', gitflow, timeout=redis_cache_timeout_project_id)
    except Exception as e:
        server_logger.error(f"Failed to update cache (gitflow_project_ids): {e}")
        verify_redis_cache()
    try:
        cache.set('trunk_project_ids', trunk, timeout=redis_cache_timeout_project_id)
    except Exception as e:
        server_logger.error(f"Failed to update cache (trunk_project_ids): {e}")
        verify_redis_cache()
    return gitflow, trunk

def get_cached_project_ids(key):
    ids = []
    try:
        ids = cache.get(key)
    except Exception as e:
        verify_redis_cache()
    if not ids:
        server_logger.info(f"Cache is missing for project ID list key ({key}) – refreshing GitLab IDs...")
        gitflow, trunk = set_project_id_cache()
        ids = gitflow if key == 'gitflow_project_ids' else trunk
    return ids

def get_cached_gitflow_project_ids():
    return get_cached_project_ids('gitflow_project_ids')

def get_cached_trunk_project_ids():
    return get_cached_project_ids('trunk_project_ids')

def get_cached_total_project_ids():
    gitflow = get_cached_project_ids('gitflow_project_ids') or []
    trunk = get_cached_project_ids('trunk_project_ids') or []
    priority = load_priority_project_ids() or {}

    total_id_dict = {"gitflow": gitflow, "trunk": trunk, "priority": priority}

    priority_ids = []
    for section in priority.values():
        # Each section is itself a dict of categories
        if isinstance(section, dict):
            for ids in section.values():
                priority_ids.extend(ids)

    # Combine, deduplicate, and filter out non-ints
    all_ids = set(gitflow + trunk + priority_ids)
    total_id_list = [id for id in all_ids if isinstance(id, int)]

    return total_id_dict, total_id_list

def refresh_project_data_cache(workflow):
    ids = get_cached_gitflow_project_ids() if workflow == 'gitflow' else get_cached_trunk_project_ids()

    def refresh(id):
        with app.app_context():
            try:
                project_key = get_project_data_cache_key(workflow, id)

                project_data = get_project_data(workflow, id, token=server_gitlab_access_token, force_refresh=True)
                time.sleep(0.2 + random.uniform(0, 0.5))  # slight delay between requests (2~5 requests/sec per worker)
                try:
                    cache.set(project_key, project_data, timeout=redis_cache_timeout_project_data)
                except Exception as e:
                    server_logger.error(f"Failed to set project data cache ({project_key}) for [{id}] : {e}")
                    verify_redis_cache()
            except Exception as e:
                server_logger.error(f"[{id}] Failed to refresh project cache: {e}")

    # One context for all threads
    with cache_refresh_lock:
        with ThreadPoolExecutor(max_workers=flask_thread_pool_executor_max_workers) as executor:    # 30 GitLab API calls in parallel per request
            futures = {executor.submit(refresh, id): id for id in ids}
            for future in as_completed(futures):
                project_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    server_logger.error(f"[{project_id}] Unexpected error in thread: {e}")

def refresh_project_metrics_cache():
    projects_dict, project_id_list = get_cached_total_project_ids()

    def refresh(id):
        with app.app_context():
            try:
                project_key = get_project_metrics_cache_key(id)

                project_metrics = get_project_metrics(id, token=server_gitlab_access_token, force_refresh=True)
                time.sleep(0.2 + random.uniform(0, 0.5))  # slight delay between requests (2~5 requests/sec per worker)
                try:
                    cache.set(project_key, project_metrics, timeout=redis_cache_timeout_project_metrics)
                except Exception as e:
                    server_logger.error(f"Failed to set project metrics cache ({project_key}) for [{id}] : {e}")
                    verify_redis_cache()
            except Exception as e:
                server_logger.error(f"[{id}] Failed to refresh project cache: {e}")

    # One context for all threads
    with cache_refresh_lock:
        with ThreadPoolExecutor(max_workers=flask_thread_pool_executor_max_workers) as executor:    # 30 GitLab API calls in parallel per request
            futures = {executor.submit(refresh, id): id for id in project_id_list}
            for future in as_completed(futures):
                project_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    server_logger.error(f"[{project_id}] Unexpected error in thread: {e}")

########## Scheduled Cache refreshes ##########

def update_project_id_cache():
    try:
        server_logger.info(f"🆔  Starting scheduled ID cache refresh at {datetime.now()}... 🔄")
        gitflow, trunk = set_project_id_cache()
        server_logger.info(f"🆔 Scheduled ID cache successfully refreshed at {datetime.now()}")
    except Exception as e:
        server_logger.error(f"Failed to update project ID cache: {e}")

def update_project_data_cache():
    try:
        server_logger.info(f"💾  Starting scheduled data cache refresh at {datetime.now()}... 🔄")
        refresh_project_data_cache('trunk')
        refresh_project_data_cache('gitflow')
        server_logger.info(f"💾  Scheduled data cache successfully refreshed at {datetime.now()}")
    except Exception as e:
        server_logger.error(f"Failed to update project data cache: {e}")

def update_project_metrics_cache():
    try:
        server_logger.info(f"📊  Starting scheduled metrics cache refresh at {datetime.now()}... 🔄")
        refresh_project_metrics_cache()
        server_logger.info(f"📊  Scheduled metrics cache successfully refreshed at {datetime.now()}")
    except Exception as e:
        server_logger.error(f"Failed to update project metrics cache: {e}")

def background_executor_with_timeout(update_function, timeout=600):
    # This function runs update_function in a separate process and enforces a timeout (in seconds).
    p = multiprocessing.Process(target=update_function)
    p.start()
    p.join(timeout)
    if p.is_alive():
        server_logger.error(f"'{update_function.__name__}' timed out after '{timeout}' seconds.")
        p.terminate()
        p.join()

# persistent cache, refreshes once a day by scheduler
def project_id_scheduler():
    scheduler = BackgroundScheduler()
    # process timeout in 10 min
    scheduler.add_job(lambda: background_executor_with_timeout(update_project_id_cache, 600), 'interval', days=1)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))

# non-persistent cache, refreshes every 10 min, delayed by 10 sec to avoid overloading GitLab API with project_metrics_scheduler requests
def project_data_scheduler():
    scheduler = BackgroundScheduler()
    # process timeout in 9 min 30 sec
    scheduler.add_job(lambda: background_executor_with_timeout(update_project_data_cache, 270), 'interval', minutes=5, max_instances=1, misfire_grace_time=30, next_run_time=next_run_with_offset)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))

# non-persistent cache, refreshes every 15 min, delayed by 1 min to avoid overloading GitLab API with project_data_scheduler requests in addition to cache_refresh_lock
def project_metrics_scheduler():
    scheduler = BackgroundScheduler()
    # process timeout in 14 min 30 sec
    scheduler.add_job(lambda: background_executor_with_timeout(update_project_metrics_cache, 570), 'interval', minutes=10, max_instances=1, misfire_grace_time=30, next_run_time=next_run_with_offset + timedelta(minutes=1))
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown(wait=False))

##################################################################

#################### Helper Functions ############################
def fetch_latest_pipeline(project, regex_pattern):
    try:
        pipelines = fetch_with_retries(lambda: project.pipelines.list(get_all=True, scope='branches', order_by='id', sort='desc'))
        matching_pipelines = [pipeline for pipeline in pipelines if regex_pattern.match(pipeline.ref)]
        return max(matching_pipelines, key=lambda p: parse(p.created_at)) if matching_pipelines else {}
    except gitlab.exceptions.GitlabListError as e:
        app_logger.info(f"INFO: Unable to fetch latest pipeline ({regex_pattern}) for project {project.name} ({project.id}): {e}", extra={'context': 'session'})
        return {}

def fetch_latest_branch(project, regex_pattern):
    try:
        branches = fetch_with_retries(lambda: project.branches.list(get_all=True))
        matching_branches = [branch for branch in branches if regex_pattern.match(branch.name)]
        return max(matching_branches, key=lambda p: parse(p.commit['committed_date'])) if matching_branches else {}
    except gitlab.exceptions.GitlabListError as e:
        app_logger.info(f"INFO: Unable to fetch latest branch ({regex_pattern}) for project {project.name} ({project.id}): {e}", extra={'context': 'session'})
        return {}

def fetch_latest_tag(workflow, project, version):
    try:
        match workflow:
            case "gitflow":
                tag = fetch_with_retries(lambda: project.tags.get(f'r{version}'))
                return tag.name.replace('r', '')
            case "trunk":
                tag = fetch_with_retries(lambda: project.tags.get(f'{version}'))
                return tag.name
            case _:
                app_logger.info(f"INFO: Unable to fetch latest tag ({version}) for project {project.name} ({project.id}) for undefined Git Workflow - {workflow}", extra={'context': 'session'})
                return None
    except gitlab.exceptions.GitlabGetError as e:
        app_logger.info(f"INFO: Unable to fetch latest tag ({version}) for project {project.name} ({project.id}): {e}")
        return None

# Fetching the latest release deployment time using environments
def get_project_environments(project):
    last_deployment_stage = None
    last_deployment_prod = None

    try:
        environments = fetch_with_retries(lambda: project.environments.list(get_all=True))
        environment_ids = {env.name: env.id for env in environments}

        stage_env_id = environment_ids.get('stage') if environment_ids.get('stage') else environment_ids.get('staging') if environment_ids.get('staging') else None
        prod_env_id = environment_ids.get('prod') if environment_ids.get('prod') else environment_ids.get('production') if environment_ids.get('production') else None

        if stage_env_id:
            stage_environment = fetch_with_retries(lambda: project.environments.get(stage_env_id))
            if stage_environment.last_deployment:
                last_deployment_stage = stage_environment.last_deployment

        if prod_env_id:
            prod_environment = fetch_with_retries(lambda: project.environments.get(prod_env_id))
            if prod_environment.last_deployment:
                last_deployment_prod = prod_environment.last_deployment

    except gitlab.exceptions.GitlabListError as e:
        app_logger.info(f"INFO: Unable to fetch latest environment deployments for project {project.name} ({project.id}): {e}")

    return last_deployment_stage, last_deployment_prod

def env_name_to_id_map(project):
    envs = fetch_with_retries(lambda: project.environments.list(get_all=True))
    return {e.name: e.id for e in envs}

# Return (deployment, pipeline) for the latest SUCCESS deploy to any env in candidates on the given ref
def latest_successful_deploy_on_release_env(project, ref, env_name_candidates):
    env_map = env_name_to_id_map(project)
    env_ids = [env_map.get(n) for n in env_name_candidates if env_map.get(n)]
    if not env_ids:
        return None, None

    # Get latest deployment and pipeline
    for env_id in env_ids:
        env = fetch_with_retries(lambda: project.environments.get(env_id))
        if getattr(env, 'state', None) == 'available':
            deployments = fetch_with_retries(lambda: project.deployments.list(environment=env.name, ref=f'release-{ref}', status='success', order_by='id', sort='desc', per_page=1, get_all=False))
            if deployments:
                d = fetch_with_retries(lambda: project.deployments.get(deployments[0].id))
                pl = fetch_with_retries(lambda: project.pipelines.get(d.deployable['pipeline']['id'])) if d.deployable and 'pipeline' in d.deployable else None
                return d, pl
    return None, None

def find_branch_created_at(project, branch_name):
    try:
        branch = fetch_with_retries(lambda: project.branches.get(branch_name))
        sha = branch.commit['id']
        c = fetch_with_retries(lambda: project.commits.get(sha))
        return c.committed_date
    except gitlab.exceptions.GitlabGetError:
        return None

def find_job_created_at(project, branch_name, job_names):
    try:
        pipelines = fetch_with_retries(lambda: project.pipelines.list(ref=branch_name, order_by='id', sort='desc', per_page=1, get_all=False))
        if not pipelines:
            return None
        pipeline_full = fetch_with_retries(lambda: project.pipelines.get(pipelines[0].id))
        jobs = fetch_with_retries(lambda: pipeline_full.jobs.list(get_all=True))
        for job in jobs:
            if job.name in job_names and job.status == 'success':
                return job.started_at
    except gitlab.exceptions.GitlabGetError:
        return None

# Return recent MRs merged into default branch (or commits if you don’t always use MRs)
def find_merge_to_default_branch_since(project, default_branch, since_iso, until_iso, limit=50):
    try:
        mrs = fetch_with_retries(lambda: project.mergerequests.list(
            get_all=True, state='merged', target_branch=default_branch,
            updated_after=since_iso, updated_before=until_iso,
            order_by='updated_at', sort='desc', per_page=limit
        ))
        return mrs
    except Exception as e:
        app_logger.info(f"INFO: Unable to fetch default branch mrs for project {project.name} ({project.id}): {e}", extra={'context': 'session'})
        return []

def pipeline_by_sha(project, sha):
    pls = fetch_with_retries(lambda: project.pipelines.list(sha=sha, order_by='id', sort='desc', per_page=1, get_all=False))
    return pls[0] if pls else None

def job_by_name(pipeline, job_names):
    jobs = fetch_with_retries(lambda: pipeline.jobs.list(get_all=True))
    for j in jobs:
        if j.name in job_names:
            return j
    return None

def find_tag_by_commit_sha(project, target_sha):
    tags = fetch_with_retries(lambda: project.tags.list(get_all=True))
    for tag in tags:
        if tag.commit['id'] == target_sha:
            return tag
    return None

def iso_to_dt(s):
    from dateutil.parser import parse
    return parse(s) if s else None

def dt_seconds(a, b):
    if a and b:
        delta = (b - a).total_seconds()
        if delta < 0:
            app_logger.info(f"INFO: Negative duration detected: start={a}, end={b}. Returning None.")
            return None
        return delta
    return None

def get_latest_commit_to_dev_lead_time(project, deploy_dev_jobs, repo_type='gitflow'):
    """
    Returns the (lead_seconds, commit_sha, latest_dev_ref.
    repo_type: 'gitflow' for develop-as-main, 'trunk' for release branch pattern.
    """
    if repo_type == 'gitflow':
        # Use default branch; get latest pipeline with a dev deploy job
        default_branch = project.default_branch
        pipelines = fetch_with_retries(lambda: project.pipelines.list(ref=default_branch, order_by='id', sort='desc', per_page=10, get_all=False))

        for pl in pipelines:
            pipeline_full = fetch_with_retries(lambda: project.pipelines.get(pl.id))
            latest_dev_ref = pipeline_full.ref
            jobs = fetch_with_retries(lambda: pipeline_full.jobs.list(get_all=True))
            for job in jobs:
                if job.name in deploy_dev_jobs and job.status == 'success':
                    commit_sha = pipeline_full.sha
                    commit_short_sha = project.commits.get(commit_sha).short_id
                    commit_obj = project.commits.get(commit_sha)
                    commit_time = iso_to_dt(commit_obj.committed_date)
                    deploy_finished_at = iso_to_dt(job.finished_at)

                    if commit_time and deploy_finished_at:
                        secs = dt_seconds(commit_time, deploy_finished_at)
                        if secs is not None:
                            return secs, commit_short_sha, latest_dev_ref
        return None, None, None
    elif repo_type == 'trunk':
        # Find the latest release branch
        import re
        branch_pattern = re.compile(r'^release-[0-9]+\.[0-9]+\.[0-9]+$')
        branches = fetch_with_retries(lambda: project.branches.list(get_all=True))
        release_branches = [b for b in branches if branch_pattern.match(b.name)]
        if not release_branches:
            return None, None, None

        # Get the most recently created release branch
        latest_release = max(release_branches, key=lambda b: iso_to_dt(b.commit['committed_date']))
        release_name = latest_release.name   # Eg: 'release-1.2.3'
        release_sha = latest_release.commit['id']

        commit_obj = fetch_with_retries(lambda: project.commits.get(release_sha))
        commit_time = iso_to_dt(commit_obj.committed_date)

        # Merge commit (on default branch) is the immediate parent commit of the release branch commit HEAD
        parent_shas = fetch_with_retries(lambda: commit_obj.parent_ids)
        merge_commit_sha = parent_shas[0] if parent_shas else None
        release_short_sha = fetch_with_retries(lambda: project.commits.get(merge_commit_sha).short_id)

        # Find the latest successful dev deploy job on the release branch
        pipelines = fetch_with_retries(lambda: project.pipelines.list(ref=release_name, order_by='id', sort='desc', per_page=1, get_all=False))
        if not pipelines:
            return None, None, None
        pipeline_full = fetch_with_retries(lambda: project.pipelines.get(pipelines[0].id))
        latest_dev_ref = pipeline_full.ref.replace('release-', '') + '-SNAPSHOT'
        jobs = fetch_with_retries(lambda: pipeline_full.jobs.list(get_all=True))
        for job in jobs:
            if job.name in deploy_dev_jobs and job.status == 'success':
                deploy_finished_at = iso_to_dt(job.finished_at)
                if commit_time and deploy_finished_at:
                    secs = dt_seconds(commit_time, deploy_finished_at)
                    if secs is not None:
                        return secs, release_short_sha, latest_dev_ref
        return None, None, None
    else:
        raise ValueError("repo_type must be 'gitflow' or 'trunk'")

def get_last_10_release_pipelines(project):
    # Fetch more than 10 pipelines to ensure we have enough after filtering
    pipelines = fetch_with_retries(lambda: project.pipelines.list(order_by='id', sort='desc', per_page=50, get_all=False))
    release_pattern = re.compile(r'^release-[0-9]+\.[0-9]+\.[0-9]+$')
    # Filter only release-* pipelines
    release_pipelines = [pl for pl in pipelines if release_pattern.match(pl.ref)]
    # Pick the last 10
    return release_pipelines[:10]

def get_project_metrics(project_id, token=None, force_refresh=False):
    # If token not provided, fallback to session (interactive access)
    if not token and 'gitlab_token' in session:
        token = session['gitlab_token'][0]
    elif not token:
        raise ValueError("GitLab Data: Server access token not provided and user access token not found in session")

    try:
        # Redis cache check
        cache_key = get_project_metrics_cache_key(project_id)
        if not force_refresh:
            try:
                cached_data = cache.get(cache_key)
            except Exception as e:
                server_logger.error(f"Failed to get cache ({cache_key}): {e}")
                verify_redis_cache()
            if cached_data:
                log_request_context(f"📊  CACHED: successfully fetched project metrics for ({project_id})")
                return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
            else:
                return {}

        try:
            gl = get_gitlab_client(token)
            project = fetch_with_retries(lambda: gl.projects.get(project_id))
        except (gitlab.exceptions.GitlabConnectionError, requests.exceptions.ConnectTimeout) as e:
            server_logger.error(f'INFO: Unable to fetch project data for project ID {project_id}: {e}')
            return {}

        project_id = project.id
        project_name = project.name
        default_branch = project.default_branch

        last_deployment_stage, last_deployment_prod = get_project_environments(project)
        latest_stage_version = last_deployment_stage['ref'].replace('release-', '') if last_deployment_stage else None
        latest_prod_version = last_deployment_prod['ref'].replace('release-', '') if last_deployment_prod else None

        release_pipelines = get_last_10_release_pipelines(project)

        dev_envs   = ['dev', 'development']
        stage_envs = ['stage', 'staging']
        prod_envs  = ['prod', 'production']
        deploy_dev_jobs = ['deploy:dev', 'apply:dev']
        deploy_stage_jobs = ['deploy:stage', 'apply:stage']
        deploy_prod_jobs = ['deploy:prod', 'apply:prod']
        release_creation_jobs = ['release:create']
        release_approval_jobs = ['release:approval']

        # 1) Commit→Dev deploy lead time (median over last N merges) from the last 14 days window
        if default_branch == 'develop':
            latest_dev_lead_secs, latest_dev_commit_sha, latest_dev_ref = get_latest_commit_to_dev_lead_time(project, deploy_dev_jobs, 'gitflow')
        else:
            latest_dev_lead_secs, latest_dev_commit_sha, latest_dev_ref = get_latest_commit_to_dev_lead_time(project, deploy_dev_jobs, 'trunk')

        until = datetime.utcnow()
        since = until - timedelta(days=14)
        mrs = find_merge_to_default_branch_since(project, default_branch, since.isoformat(), until.isoformat(), limit=60)
        commit_to_dev_secs = []
        for mr in mrs:
            merged_at = iso_to_dt(mr.merged_at)
            # Use the merge commit SHA when possible; fall back to last commit on MR.
            sha = mr.merge_commit_sha or (mr.sha if hasattr(mr, 'sha') else None)
            if not merged_at or not sha:
                continue

            # Find latest successful dev deployment for the ref used by that pipeline
            if default_branch == 'develop':
                pl = pipeline_by_sha(project, sha)
            else:
                trunk_snapshot_tag = find_tag_by_commit_sha(project, sha)
                if not trunk_snapshot_tag:
                    continue
                trunk_dev_ref = 'release-' + trunk_snapshot_tag.name.replace('-SNAPSHOT', '')
                pls = fetch_with_retries(lambda: project.pipelines.list(ref=trunk_dev_ref, order_by='id', sort='desc', per_page=1, get_all=False))
                if not pls:
                    continue
                pl = fetch_with_retries(lambda: project.pipelines.get(pls[0].id))

            if not pl:
                continue
            dev_deploy = None
            jobs = fetch_with_retries(lambda: pl.jobs.list(get_all=True))
            for j in jobs:
                job_name = j.name
                if job_name and job_name == 'deploy:dev':
                    dev_deploy = j
                    break
            finished_at = iso_to_dt(getattr(dev_deploy, 'finished_at', None))
            secs = dt_seconds(merged_at, finished_at)
            if secs:
                commit_to_dev_secs.append(secs)

        commit_to_dev_median_secs = (sorted(commit_to_dev_secs)[len(commit_to_dev_secs)//2] if commit_to_dev_secs else None)

        # 2) Release creation → Stage deploy lead time
        if default_branch == 'develop':
            rel_created_at = iso_to_dt(find_branch_created_at(project, f"release-{latest_stage_version}")) if latest_stage_version else None
        else:
            rel_created_at = iso_to_dt(find_job_created_at(project, f"release-{latest_stage_version}", release_creation_jobs)) if latest_stage_version else None
        stage_deployment, stage_pl = latest_successful_deploy_on_release_env(project, latest_stage_version, stage_envs) if latest_stage_version else (None, None)
        stage_deployed_at = iso_to_dt(getattr(stage_deployment, 'finished_at', None) or getattr(stage_deployment, 'updated_at', None))
        latest_release_to_stage_secs = dt_seconds(rel_created_at, stage_deployed_at)

        release_to_stage_secs = []
        for pl in release_pipelines:
            if default_branch == 'develop':
                release_started_at = iso_to_dt(find_branch_created_at(project, pl.ref))
            else:
                release_started_at = iso_to_dt(find_job_created_at(project, pl.ref, release_creation_jobs))
            stage_deploy = None
            jobs = fetch_with_retries(lambda: pl.jobs.list(get_all=True))
            for j in jobs:
                job_name = j.name
                if job_name and job_name in deploy_stage_jobs:
                    stage_deploy = j
                    break
            finished_at = iso_to_dt(getattr(stage_deploy, 'finished_at', None))
            secs = dt_seconds(release_started_at, finished_at)
            if secs:
                release_to_stage_secs.append(secs)

        release_to_stage_median_secs = (sorted(release_to_stage_secs)[len(release_to_stage_secs)//2] if release_to_stage_secs else None)

        # 3) Prod approval → Prod deploy lead time
        prod_deployment, prod_pl = latest_successful_deploy_on_release_env(project, latest_prod_version, prod_envs) if latest_prod_version else (None, None)
        prod_deployed_at = iso_to_dt(getattr(prod_deployment, 'finished_at', None) or getattr(prod_deployment, 'updated_at', None))
        prod_approval_at = None
        if prod_pl:
            approve_job = job_by_name(prod_pl, release_approval_jobs)
            if approve_job and approve_job.status == 'success':
                prod_approval_at = iso_to_dt(approve_job.finished_at)
        latest_approval_to_prod_secs = dt_seconds(prod_approval_at, prod_deployed_at)

        release_to_prod_secs = []
        for pl in release_pipelines:
            release_approval = None
            prod_deploy = None
            jobs = fetch_with_retries(lambda: pl.jobs.list(get_all=True))
            for j in jobs:
                job_name = j.name
                if job_name in release_approval_jobs:
                    release_approval = j
                elif job_name in deploy_prod_jobs:
                    prod_deploy = j

            # Only compute if both jobs are found
            if release_approval and prod_deploy:
                approval_at = iso_to_dt(getattr(release_approval, 'started_at', None))
                finished_at = iso_to_dt(getattr(prod_deploy, 'finished_at', None))
                secs = dt_seconds(approval_at, finished_at)
                if secs is not None:
                    release_to_prod_secs.append(secs)

        approval_to_prod_median_secs = (sorted(release_to_prod_secs)[len(release_to_prod_secs)//2] if release_to_prod_secs else None)

        # 4) Pipeline duration / success rate (last N pipelines on default branch)
        pls = fetch_with_retries(lambda: project.pipelines.list(ref=default_branch, order_by='id', sort='desc', per_page=50, get_all=False))
        durations = []
        success = 0
        for pl in pls:
            pl_full = fetch_with_retries(lambda: project.pipelines.get(pl.id))
            if pl_full.status == 'success':
                success += 1
            if getattr(pl_full, 'duration', None):
                durations.append(pl_full.duration)
            else:
                s = iso_to_dt(pl_full.started_at); f = iso_to_dt(pl_full.finished_at)
                if s and f:
                    durations.append((f - s).total_seconds())
        pipeline_duration_median_secs = (sorted(durations)[len(durations)//2] if durations else None)
        pipeline_success_rate = (success / len(pls)) if pls else None

        response = {
            'project_id': project_id,
            'metrics': {
                'latest_dev_ref': latest_dev_ref,
                'latest_dev_commit_sha': latest_dev_commit_sha,
                'latest_dev_lead_secs': latest_dev_lead_secs,
                'commit_to_dev_median_secs': commit_to_dev_median_secs,
                'latest_stage_version': latest_stage_version,
                'latest_release_to_stage_secs': latest_release_to_stage_secs,
                'release_to_stage_median_secs': release_to_stage_median_secs,
                'latest_prod_version': latest_prod_version,
                'latest_approval_to_prod_secs': latest_approval_to_prod_secs,
                'approval_to_prod_median_secs': approval_to_prod_median_secs,
                'pipeline_duration_median_secs': pipeline_duration_median_secs,
                'pipeline_success_rate': pipeline_success_rate,
            },
            'metrics_timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        try:
            cache.set(cache_key, response, timeout=redis_cache_timeout_project_metrics)
        except Exception as e:
            server_logger.error(f"Failed to update cache ({cache_key}): {e}")
            verify_redis_cache()

        log_request_context(f"📊  GITLAB: successfully fetched project metrics for {project_name} ({project_id})")
        return response
    except Exception as e:
        log_error_context(f'Error fetching project metrics for project ID {project_id}: {e}')
        return {}

def get_project_data(workflow, project_id, token=None, force_refresh=False):
    # If token not provided, fallback to session (interactive access)
    if not token and 'gitlab_token' in session:
        token = session['gitlab_token'][0]
    elif not token:
        raise ValueError("GitLab Data: Server access token not provided and user access token not found in session")

    try:
        # Redis cache check
        cache_key = get_project_data_cache_key(workflow, project_id)
        if not force_refresh:
            try:
                cached_data = cache.get(cache_key)
            except Exception as e:
                server_logger.error(f"Failed to get cache ({cache_key}): {e}")
                verify_redis_cache()
            if cached_data:
                log_request_context(f"💾  CACHED: successfully fetched project data for {project_id}")
                return jsonify(cached_data) if isinstance(cached_data, dict) else jsonify(json.loads(cached_data)) if isinstance(cached_data, str) else cached_data
            else:
                return jsonify({'error': 'No cached data found'}), 404


        try:
            gl = get_gitlab_client(token)
            project = fetch_with_retries(lambda: gl.projects.get(project_id))
        except (gitlab.exceptions.GitlabConnectionError, requests.exceptions.ConnectTimeout) as e:
            server_logger.error(f'INFO: Unable to fetch project data for project ID {project_id}: {e}')
            return None

        project_id = project.id
        project_name = project.name
        project_url = project.web_url
        last_activity_at = project.last_activity_at

        last_deployment_stage, last_deployment_prod = get_project_environments(project)
        last_deployment_stage_at = last_deployment_stage.get('updated_at') if last_deployment_stage else None
        last_deployment_prod_at = last_deployment_prod.get('updated_at') if last_deployment_prod else None
        latest_stage_deployment = last_deployment_stage['ref'].replace('release-', '') if last_deployment_stage else None
        latest_prod_deployment = last_deployment_prod['ref'].replace('release-', '') if last_deployment_prod else None

        # Determine last_release_deployment_at
        if last_deployment_stage_at and last_deployment_prod_at:
            last_release_deployment_at = max(last_deployment_stage_at, last_deployment_prod_at, key=parse)
        elif last_deployment_stage_at:
            last_release_deployment_at = last_deployment_stage_at
        elif last_deployment_prod_at:
            last_release_deployment_at = last_deployment_prod_at
        else:
            last_release_deployment_at = None

        project_metrics = get_project_metrics(project_id, token)

        response = {}
        match workflow:
            case "gitflow":
                last_stage_deployment_at = last_deployment_stage.get('updated_at') if last_deployment_stage else None
                last_prod_deployment_at = last_deployment_prod.get('updated_at') if last_deployment_prod else None

                ## Latest Release Branch and Pipeline Statuses
                latest_release_pipeline = fetch_latest_pipeline(project, re.compile(r'^release-[0-9]+\.[0-9]+\.0$'))
                latest_release_pipeline_version = latest_release_pipeline.ref.replace('release-', '') if latest_release_pipeline else None
                latest_release_pipeline_url = latest_release_pipeline.web_url if latest_release_pipeline else None
                latest_release_pipeline_status = latest_release_pipeline.status if latest_release_pipeline else None
                # if latest_release_pipeline_status == "manual":
                #     latest_release_pipeline_jobs = latest_release_pipeline.jobs.list()
                #     latest_release_pipeline_status = ', '.join([job.status for job in latest_release_pipeline_jobs if job.name == 'deploy:prod'])


                latest_release_branch = fetch_latest_branch(project, re.compile(r'^release-[0-9]+\.[0-9]+\.0$'))
                latest_release_branch_version = latest_release_branch.name.replace('release-', '') if latest_release_branch else None

                latest_release_branch_tag_version = fetch_latest_tag(workflow, project, latest_release_branch_version)
                latest_release_pipeline_tag_version = fetch_latest_tag(workflow, project, latest_release_pipeline_version)

                ## Latest Hotfix Branch and Pipeline Statuses
                latest_hotfix_pipeline = fetch_latest_pipeline(project, re.compile(r'^release-[0-9]+\.[0-9]+\.[1-9]+$'))
                latest_hotfix_pipeline_version = latest_hotfix_pipeline.ref.replace('release-', '') if latest_hotfix_pipeline else None
                latest_hotfix_pipeline_url = latest_hotfix_pipeline.web_url if latest_hotfix_pipeline else None
                latest_hotfix_pipeline_status = latest_hotfix_pipeline.status if latest_hotfix_pipeline else None

                latest_hotfix_branch = fetch_latest_branch(project, re.compile(r'^release-[0-9]+\.[0-9]+\.[1-9]+$'))
                latest_hotfix_branch_version = latest_hotfix_branch.name.replace('release-', '') if latest_hotfix_branch else None

                latest_hotfix_branch_tag_version = fetch_latest_tag(workflow, project, latest_hotfix_branch_version)
                latest_hotfix_pipeline_tag_version = fetch_latest_tag(workflow, project, latest_hotfix_pipeline_version)

                current_timestamp = datetime.utcnow().isoformat() + 'Z'  # ISO 8601 format in UTC

                response = {
                    'workflow': workflow,
                    'project_id': project_id,
                    'project_name': project_name,
                    'project_url': project_url,
                    'last_activity_at': last_activity_at,
                    'last_release_deployment_at': last_release_deployment_at,
                    'last_stage_deployment_at': last_stage_deployment_at,
                    'last_prod_deployment_at': last_prod_deployment_at,
                    'latest_stage_deployment': latest_stage_deployment,
                    'latest_prod_deployment': latest_prod_deployment,
                    'latest_release_branch_version': latest_release_branch_version,
                    'latest_release_branch_tag_version': latest_release_branch_tag_version,
                    'latest_release_pipeline_version': latest_release_pipeline_version,
                    'latest_release_pipeline_tag_version': latest_release_pipeline_tag_version,
                    'latest_release_pipeline_status': latest_release_pipeline_status,
                    'latest_release_pipeline_url': latest_release_pipeline_url,
                    'latest_hotfix_branch_version': latest_hotfix_branch_version,
                    'latest_hotfix_branch_tag_version': latest_hotfix_branch_tag_version,
                    'latest_hotfix_pipeline_version': latest_hotfix_pipeline_version,
                    'latest_hotfix_pipeline_tag_version': latest_hotfix_pipeline_tag_version,
                    'latest_hotfix_pipeline_status': latest_hotfix_pipeline_status,
                    'latest_hotfix_pipeline_url': latest_hotfix_pipeline_url,
                    'last_refreshed_at': current_timestamp,
                    'metrics': {
                        'latest_dev_ref': project_metrics.get('metrics', {}).get('latest_dev_ref', None),
                        'latest_dev_commit_sha': project_metrics.get('metrics', {}).get('latest_dev_commit_sha', None),
                        'latest_dev_lead_secs': project_metrics.get('metrics', {}).get('latest_dev_lead_secs', None),
                        'commit_to_dev_median_secs': project_metrics.get('metrics', {}).get('commit_to_dev_median_secs', None),
                        'latest_stage_version': project_metrics.get('metrics', {}).get('latest_stage_version', None),
                        'latest_release_to_stage_secs': project_metrics.get('metrics', {}).get('latest_release_to_stage_secs', None),
                        'release_to_stage_median_secs': project_metrics.get('metrics', {}).get('release_to_stage_median_secs', None),
                        'latest_prod_version': project_metrics.get('metrics', {}).get('latest_prod_version', None),
                        'latest_approval_to_prod_secs': project_metrics.get('metrics', {}).get('latest_approval_to_prod_secs', None),
                        'approval_to_prod_median_secs': project_metrics.get('metrics', {}).get('approval_to_prod_median_secs', None),
                        'pipeline_duration_median_secs': project_metrics.get('metrics', {}).get('pipeline_duration_median_secs', None),
                        'pipeline_success_rate': project_metrics.get('metrics', {}).get('pipeline_success_rate', None)
                    }
                }
            case "trunk":

                ## Latest Release Branch and Pipeline Statuses
                latest_release_pipeline = fetch_latest_pipeline(project, re.compile(r'^release-[0-9]+\.[0-9]+\.[0-9]+$'))
                latest_release_pipeline_version = latest_release_pipeline.ref.replace('release-', '') if latest_release_pipeline else None
                latest_release_pipeline_url = latest_release_pipeline.web_url if latest_release_pipeline else None
                latest_release_pipeline_status = latest_release_pipeline.status if latest_release_pipeline else None

                latest_release_branch = fetch_latest_branch(project, re.compile(r'^release-[0-9]+\.[0-9]+\.[0-9]+$'))
                latest_release_branch_version = latest_release_branch.name.replace('release-', '') if latest_release_branch else None

                latest_release_branch_tag_version = fetch_latest_tag(workflow, project, latest_release_branch_version)
                latest_release_pipeline_tag_version = fetch_latest_tag(workflow, project, latest_release_pipeline_version)

                last_stage_deployment_at = last_deployment_stage.get('updated_at') if last_deployment_stage else None
                last_prod_deployment_at = last_deployment_prod.get('updated_at') if last_deployment_prod else None

                current_timestamp = datetime.utcnow().isoformat() + 'Z'  # ISO 8601 format in UTC

                response = {
                    'workflow': workflow,
                    'project_id': project_id,
                    'project_name': project_name,
                    'project_url': project_url,
                    'last_activity_at': last_activity_at,
                    'last_release_deployment_at': last_release_deployment_at,
                    'last_stage_deployment_at': last_stage_deployment_at,
                    'last_prod_deployment_at': last_prod_deployment_at,
                    'latest_stage_deployment': latest_stage_deployment,
                    'latest_prod_deployment': latest_prod_deployment,
                    'latest_release_branch_version': latest_release_branch_version,
                    'latest_release_branch_tag_version': latest_release_branch_tag_version,
                    'latest_release_pipeline_version': latest_release_pipeline_version,
                    'latest_release_pipeline_tag_version': latest_release_pipeline_tag_version,
                    'latest_release_pipeline_status': latest_release_pipeline_status,
                    'latest_release_pipeline_url': latest_release_pipeline_url,
                    'last_refreshed_at': current_timestamp,
                    'metrics': {
                        'latest_dev_ref': project_metrics.get('metrics', {}).get('latest_dev_ref', None),
                        'latest_dev_commit_sha': project_metrics.get('metrics', {}).get('latest_dev_commit_sha', None),
                        'latest_dev_lead_secs': project_metrics.get('metrics', {}).get('latest_dev_lead_secs', None),
                        'commit_to_dev_median_secs': project_metrics.get('metrics', {}).get('commit_to_dev_median_secs', None),
                        'latest_stage_version': project_metrics.get('metrics', {}).get('latest_stage_version', None),
                        'latest_release_to_stage_secs': project_metrics.get('metrics', {}).get('latest_release_to_stage_secs', None),
                        'release_to_stage_median_secs': project_metrics.get('metrics', {}).get('release_to_stage_median_secs', None),
                        'latest_prod_version': project_metrics.get('metrics', {}).get('latest_prod_version', None),
                        'latest_approval_to_prod_secs': project_metrics.get('metrics', {}).get('latest_approval_to_prod_secs', None),
                        'approval_to_prod_median_secs': project_metrics.get('metrics', {}).get('approval_to_prod_median_secs', None),
                        'pipeline_duration_median_secs': project_metrics.get('metrics', {}).get('pipeline_duration_median_secs', None),
                        'pipeline_success_rate': project_metrics.get('metrics', {}).get('pipeline_success_rate', None)
                    }
                }
            case _:
                app_logger.error(f"Undefined Git Workflow - {workflow}")

        try:
            cache.set(cache_key, response, timeout=redis_cache_timeout_project_data)
        except Exception as e:
            server_logger.error(f"Failed to update cache ({cache_key}): {e}")
            verify_redis_cache()

        log_request_context(f"💾  GITLAB: successfully fetched project data for {project_name} ({project_id})")
        return jsonify(response)
    except Exception as e:
        log_error_context(f'Error fetching project data for project ID {project_id}: {e}')
        return jsonify({'error': str(e)}), 500

def hotfix_operations(project_id, version, operation, token=None):
    # If token not provided, fallback to session (interactive access)
    if not token and 'gitlab_token' in session:
        token = session['gitlab_token'][0]
    elif not token:
        raise ValueError("GitLab Data: Server access token not provided and user access token not found in session")

    try:
        gl = get_gitlab_client(token)
        gl.auth()
        current_user_id = gl.user.id
        current_user_name = gl.user.name
        project = fetch_with_retries(lambda: gl.projects.get(project_id))

        member = fetch_with_retries(lambda: project.members_all.get(current_user_id))
        access_level = member.access_level
    except GitlabAuthenticationError:
        # clear the bad token so next time user is forced to re-login
        session.pop('gitlab_token', None)
        return jsonify({'error': 'invalid_token'}), 401
    except (gitlab.exceptions.GitlabConnectionError, requests.exceptions.ConnectTimeout) as e:
        server_logger.error(f'INFO: Unable to fetch project data for project ID {project_id}: {e}')
        return None

    latest_prod_ref = f"release-{version}"
    next_hotfix_version = None
    next_hotfix_branch = None
    hotfix_branch_exists = None
    if version:
        M, m, p = map(int, version.split('.'))
        next_hotfix_version = f"{M}.{m}.{p + 1}"
        next_hotfix_branch = f"release-{next_hotfix_version}"
    pipelines = project.pipelines.list(ref=latest_prod_ref, order_by='id', sort='desc', per_page=1, get_all=False)
    if not pipelines:
        server_logger.info(f"No pipelines found for ref {latest_prod_ref}")
        return jsonify({
            'error': "No pipelines found for provided ref",
            'job_url': project.web_url
        }), 404

    pipeline = fetch_with_retries(lambda: project.pipelines.get(pipelines[0].id))
    pipeline_job = next((j for j in fetch_with_retries(lambda: pipeline.jobs.list()) if j.name == f'hotfix:{operation}'), None)
    if not pipeline_job:
        server_logger.error(f"No 'hotfix:{operation}' job found in pipeline {pipeline.id}")
        return jsonify({
            'error': f"No 'hotfix:{operation}' job found",
            'job_url': pipeline.web_url
        }), 404

    job = fetch_with_retries(lambda: project.jobs.get(pipeline_job.id))

    if operation == "create":
        if access_level >= const.DEVELOPER_ACCESS:
            if job.status == "manual" or job.status == "canceled":
                try:
                    if job.status == "manual":
                        job.play()
                    else:
                        job.retry()
                    return jsonify({
                        'status': 'Hotfix creation is in progress...',
                        'job_url': job.web_url,
                        'hotfix_branch': next_hotfix_branch
                    })
                except requests.RequestException as e:
                    server_logger.error(f'ERROR: Pipeline trigger failed for job ({job.id}): {e}')
                    return jsonify({
                        'error': "Unable to trigger 'hotfix:create' job",
                        'job_url': job.web_url,
                        'hotfix_branch': next_hotfix_branch
                    }), 502
            elif job.status == "running":
                server_logger.error(f'INFO: Skipping "hotfix:create" job ({job.id}) for {project_id} since the job has just got triggered')
                return jsonify({
                    'status': 'Hotfix creation is already in progress',
                    'job_url': job.web_url,
                    'hotfix_branch': next_hotfix_branch
                })
            elif job.status == "success":
                if next_hotfix_branch:
                    hotfix_branch_exists = bool(fetch_with_retries(lambda: project.branches.list(search=next_hotfix_branch)))
                if hotfix_branch_exists:
                    server_logger.error(f'INFO: Skipping "hotfix:create" job ({job.id}) for {project_id} since the job was already run')
                    return jsonify({
                        'status': 'Hotfix was already created',
                        'job_url': job.web_url,
                        'hotfix_branch': next_hotfix_branch
                    })
                else:
                    server_logger.error(f'ERROR: Unable to identify "hotfix:create" job ({job.id}) status for {project_id}. Please trigger manually, if needed')
                    return jsonify({
                        'error': f'Hotfix creation job was already run, but no {next_hotfix_branch} branch exists',
                        'job_url': job.web_url,
                        'hotfix_branch': next_hotfix_branch
                    })
            else:
                server_logger.error(f'ERROR: Unable to identify "hotfix:create" job ({job.id}) status for {project_id}. Please trigger manually, if needed')
                return jsonify({
                    'error': f'Unable to create Hotfix ({job.status})',
                    'job_url': job.web_url,
                    'hotfix_branch': next_hotfix_branch
                }), 502
        else:
            server_logger.error(f'ERROR: {current_user_name} is not authorized to trigger "hotfix:create" job ({job.id}) in {project_id}')
            return jsonify({
                'error': f'{current_user_name} [{access_level}] must have DEVELOPER access to trigger job',
                'job_url': job.web_url,
                'hotfix_branch': next_hotfix_branch
            }), 401
    elif operation == "cleanup":
        if access_level >= const.MAINTAINER_ACCESS:
            if job.status == "manual" or job.status == "canceled" or job.status == "failed":
                try:
                    url = f"{gitlab_base_url}/api/v4/projects/{project_id}/jobs/{job.id}/play"
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "job_variables_attributes": [
                            {"key": "HOTFIX_CLEANUP_VERSION", "value": next_hotfix_version}
                        ]
                    }
                    resp = requests.post(url, headers=headers, json=payload, timeout=10)
                    server_logger.info(f"INFO: Successfully triggered 'hotfix:cleanup' job - {resp.json().get('web_url')}")

                    if not resp.ok:
                        server_logger.error(f'ERROR: {resp.json()}')
                        return jsonify({
                            'error': "Unable to trigger 'hotfix:cleanup' job",
                            'job_url': job.web_url,
                            'hotfix_branch': latest_prod_ref
                        }), resp.status_code

                    return jsonify({
                        'status': 'Hotfix cleanup is in progress...',
                        'job_url': job.web_url,
                        'hotfix_branch': latest_prod_ref
                    })
                except requests.RequestException as e:
                    server_logger.error(f'ERROR: Pipeline trigger failed for job ({job.id}): {e}')
                    return jsonify({
                        'error': "Unable to trigger 'hotfix:cleanup' job",
                        'job_url': job.web_url,
                        'hotfix_branch': latest_prod_ref
                    }), 502
            elif job.status == "running":
                server_logger.error(f'INFO: Skipping "hotfix:cleanup" job ({job.id}) for {project_id} since the job has just got triggered')
                return jsonify({
                    'status': 'Hotfix cleanup is already in progress',
                    'job_url': job.web_url,
                    'hotfix_branch': latest_prod_ref
                })
            elif job.status == "success":
                server_logger.error(f'INFO: Skipping "hotfix:cleanup" job ({job.id}) for {project_id} since the job was already run')
                return jsonify({
                    'status': 'Hotfix cleanup was already completed',
                    'job_url': job.web_url,
                    'hotfix_branch': latest_prod_ref
                })
            else:
                server_logger.error(f'ERROR: Unable to identify "hotfix:cleanup" job ({job.id}) status for {project_id}. Please trigger manually, if needed')
                return jsonify({
                    'error': f'Unable to cleanup Hotfix ({job.status})',
                    'job_url': job.web_url,
                    'hotfix_branch': latest_prod_ref
                }), 502
        else:
            server_logger.error(f'ERROR: {current_user_name} is not authorized to trigger "hotfix:cleanup" job ({job.id}) in {project_id}')
            return jsonify({
                'error': f'{current_user_name} [{access_level}] must have MAINTAINER access to trigger job',
                'job_url': job.web_url,
                'hotfix_branch': latest_prod_ref
            }), 401


##################################################################

# Executes automatically by Flask after every response to hard fail-safe in cache function to avoid caching bad responses
@app.after_request
def prevent_cache_on_unauthorized(response):
    if response.status_code == 401:
        app_logger.debug("401 response detected — setting Cache-Control: no-store")
        response.cache_control.no_store = True
    return response

def login_required_api(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'gitlab_token' not in session:
            return jsonify({'error': 'unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

@gitlab_auth.tokengetter
def get_gitlab_oauth_token():
    return session.get('gitlab_token')

@app.route('/login')
def login():
    return gitlab_auth.authorize(callback=url_for('authorized', _external=True))

@app.route('/logout')
def logout():
    session.pop('gitlab_token', None)
    session.clear()  # Clear the entire session
    response = make_response(redirect(url_for('login')))
    response.delete_cookie(app.session_cookie_name)  # Explicitly delete the session cookie
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/login/authorized')
def authorized():
    response = gitlab_auth.authorized_response()
    if response is None:
        error_reason = request.args.get('error_reason', 'Unknown')
        error_description = request.args.get('error_description', 'Unknown')
        return f'Access denied: reason={error_reason} error={error_description}'

    if response.get('access_token') is None:
        print("GitLab response:", response)
        return 'Failed to obtain access token from GitLab.'
    if response is None or response.get('access_token') is None:
        return 'Access denied: reason={} error={}'.format(
            request.args['error_reason'],
            request.args['error_description']
        )

    session['gitlab_token'] = (response['access_token'], '')

    # Fetch user info
    token = session['gitlab_token'][0]
    gl = get_gitlab_client(token)
    gl.auth()
    session['gitlab_user_name'] = gl.user.name  # Store user name in session

    return redirect(url_for('index'))

@app.route('/session/check')
def check_session():
    return jsonify({'logged_in': 'gitlab_token' in session})


@app.route('/')
def index():
    if 'gitlab_token' not in session:
        return redirect(url_for('login'))
    session.permanent = True  # Make the session permanent to use the session timeout
    if has_request_context() and 'gitlab_user_name' in session:
        single_line_logger.info(f"[{session['gitlab_user_name']}] ")
    else:
        single_line_logger.info(f"[{server_user_name}] ")
    return render_template('index.html', gitflow_data=get_cached_gitflow_project_ids(), trunk_data=get_cached_trunk_project_ids())

@app.route('/health', methods=['GET'])
def health():
    app_logger.info("Health Check!")
    return '{"status" : "OK"}'

@app.route('/releases')
def releases():
    if 'gitlab_token' not in session:
        return redirect(url_for('index'))
    session.permanent = True
    if has_request_context() and 'gitlab_user_name' in session:
        single_line_logger.info(f"[{session['gitlab_user_name']}] ")
    else:
        single_line_logger.info(f"[{server_user_name}] ")
    projects_dict, projects_id_list = get_cached_total_project_ids()
    return render_template('releases.html', projects_data=projects_dict)

@app.route('/project_data/<workflow>/<int:project_id>')
@login_required_api
def project_data(workflow, project_id):
    force_refresh = request.args.get('force_refresh') == 'true'
    return get_project_data(workflow, project_id, force_refresh=force_refresh)

@app.route('/project_kpis/<workflow>/<int:project_id>')
@login_required_api
def project_kpis(workflow, project_id):
    force_refresh = request.args.get('force_refresh') == 'true'
    return get_project_kpis(workflow, project_id, force_refresh=force_refresh)

@app.route('/trigger_hotfix_create/<int:project_id>')
@login_required_api
def trigger_hotfix_create(project_id):
    version = request.args.get('version')
    if not version:
        return jsonify({'error': 'version query-param is required'}), 400
    return hotfix_operations(project_id, version, 'create')

@app.route('/trigger_hotfix_cleanup/<int:project_id>')
@login_required_api
def trigger_hotfix_cleanup(project_id):
    version = request.args.get('version')
    if not version:
        return jsonify({'error': 'version query-param is required'}), 400
    return hotfix_operations(project_id, version, 'cleanup')

###################################################################


if __name__ == '__main__':
    #######################
    ## Runs on Python Flask
    #######################
    ##    python main.py
    #######################
    print("🖥️  Refreshing project metrics cache at main server start... 🔄")
    update_project_metrics_cache()
    print("🖥️  Refreshing project data cache at main server start... 🔄")
    update_project_data_cache()

    # Runs on Production
    if os.getenv('RUN_APP_SCHEDULER'):
        app.run(host='0.0.0.0', port=3000)
    else:
        app.run(host='0.0.0.0', port=3000, debug=True)
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    if gunicorn_logger.handlers:
        print("🦄  starting gunicorn server... 🚀")
        #####################################
        ## Runs on Gunicorn - Production mode
        #####################################
        ##     gunicorn main:app \
        ##         --name=release-deployment-dashboard \
        ##         --bind=127.0.0.1:3000 \
        ##         --worker-class=gthread
        ##         --workers=2 \
        ##         --threads=5 \
        ##         --timeout=120 \
        ##         --log-level=debug
        ##         #--preload (required when not working with a scheduler in a multi worker setup, to load the config once on master before repeatedly loading the same again on all Gunicorn workers)
        #####################################
        # Running under Gunicorn: adopt Gunicorn handlers to reuse gunicorn's handlers/level for identical gunicorn output
        for logger in (server_logger, app_logger, single_line_logger):
            logger.handlers = gunicorn_logger.handlers
            logger.setLevel(gunicorn_logger.level)
            logger.propagate = False  # handled by gunicorn handlers directly
    else:
        ############################################################################################
        ## Run `python scheduler.py` to schedule project_id_scheduler() and project_data_scheduler()
        ############################################################################################
        # Fallback for standalone log handler/formatter (Eg: scheduler.py)
        pass  # Do NOT clear existing handlers
