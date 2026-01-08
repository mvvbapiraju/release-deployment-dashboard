#!/usr/bin/env python

##########################################################
## RUN COMMAND: RUN_APP_SCHEDULER=true python scheduler.py
##########################################################
from main import app, update_project_metrics_cache, update_project_data_cache, project_id_scheduler, project_data_scheduler, project_metrics_scheduler
import os, time
import logging, sys


if __name__ == '__main__':
    with app.app_context():
        print("🖥️  Starting Scheduler... ⏰")
        ## To preload caches
        if os.getenv('RUN_APP_SCHEDULER'):
            update_project_metrics_cache()
            update_project_data_cache()

        # Starts recurring flask scheduler jobs
        project_id_scheduler()  # persistent cache, refreshes once a day
        project_data_scheduler()   # volatile cache, refreshes every 5 min
        project_metrics_scheduler()   # volatile cache, refreshes every 10 min

    # Keep the scheduler process alive
    try:
        print("✅  Scheduler is running... 🚀  ⏰")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑  Scheduler is shutting down gracefully...")
