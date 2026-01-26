"""
Background scheduler for automated report generation.

Runs as a daemon thread, checking every hour (by default) for any reports
that are due to be generated. Keeps things ticking along without user intervention.
"""
import os
import threading
import time
import logging
from datetime import datetime, timezone
from typing import Optional
import yaml

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import database as db
from pdf_generator import generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scheduler')

# Global scheduler instance
_scheduler_instance: Optional['ReportScheduler'] = None
_scheduler_lock = threading.Lock()


class ReportScheduler:
    """Background scheduler that checks and runs scheduled reports."""

    def __init__(self, check_interval_minutes: int = 60, api_token: str = None):
        """
        Initialize the scheduler.

        Args:
            check_interval_minutes: How often to check for due schedules (default 60)
            api_token: Default API token to use for report generation
        """
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.api_token = api_token
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the scheduler in a background thread."""
        if self._running:
            logger.warning("Scheduler is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info(f"Scheduler started (checking every {self.check_interval // 60} minutes)")

    def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("Scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def _run_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while not self._stop_event.is_set():
            try:
                self._check_and_run_schedules()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")

            # Wait for next check interval (but check stop event frequently)
            for _ in range(self.check_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("Scheduler loop exited")

    def _check_and_run_schedules(self):
        """Check all scheduled templates and run any that are due."""
        logger.debug("Checking schedules...")
        templates = db.get_scheduled_templates()

        for template in templates:
            try:
                if self._is_due(template):
                    logger.info(f"Template '{template['name']}' is due, running...")
                    self._run_template(template)
            except Exception as e:
                logger.error(f"Error processing template {template['id']}: {e}")

    def _is_due(self, template: dict) -> bool:
        """Check if a template's schedule is due to run."""
        try:
            tz = ZoneInfo(template.get('schedule_timezone') or 'UTC')
        except Exception:
            tz = ZoneInfo('UTC')
            logger.warning(f"Invalid timezone '{template.get('schedule_timezone')}', using UTC")

        now = datetime.now(tz)
        frequency = template.get('schedule_frequency')
        target_hour = template.get('schedule_hour')

        if not frequency or target_hour is None:
            return False

        # Check if we're in the right hour
        if now.hour != target_hour:
            return False

        # Check if we already ran this hour
        last_run = template.get('schedule_last_run')
        if last_run:
            try:
                last_run_dt = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                if last_run_dt.tzinfo is None:
                    last_run_dt = last_run_dt.replace(tzinfo=ZoneInfo('UTC'))
                last_run_local = last_run_dt.astimezone(tz)

                # If we ran within the last hour, skip
                if (now - last_run_local).total_seconds() < 3600:
                    return False
            except Exception as e:
                logger.warning(f"Could not parse last_run '{last_run}': {e}")

        # Check frequency-specific conditions
        if frequency == 'daily':
            return True

        elif frequency == 'weekly':
            target_dow = template.get('schedule_day_of_week', 0)  # 0 = Monday
            return now.weekday() == target_dow

        elif frequency == 'monthly':
            target_dom = template.get('schedule_day_of_month', 1)
            return now.day == target_dom

        return False

    def _run_template(self, template: dict):
        """Execute report generation for a template."""
        template_id = template['id']

        try:
            # Parse config
            config = yaml.safe_load(template['config_yaml'])
            region = config.get('global', {}).get('region', 'au1')

            # Use provided API token
            api_token = self.api_token
            if not api_token:
                logger.error(f"No API token available for template {template_id}")
                self._record_failure(template_id, "No API token configured")
                return

            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = template['name'].replace(' ', '_').replace('/', '_')
            filename = f"{safe_name}_{timestamp}.pdf"
            filepath = os.path.join(db.get_reports_directory(), filename)

            # Ensure reports directory exists
            os.makedirs(db.get_reports_directory(), exist_ok=True)

            # Generate report
            logger.info(f"Generating report: {filename}")
            orientation = config.get('global', {}).get('orientation', 'portrait')
            success, error = generate_report(
                config=config,
                output_path=filepath,
                api_token=api_token,
                region=region,
                orientation=orientation
            )

            if success:
                file_size = os.path.getsize(filepath)
                db.create_generated_report(
                    template_id=template_id,
                    filename=filename,
                    file_path=filepath,
                    status="success",
                    file_size=file_size
                )
                logger.info(f"Report generated successfully: {filename} ({file_size} bytes)")
            else:
                self._record_failure(template_id, error, filename, filepath)

            # Update last_run
            db.update_template(template_id, schedule_last_run=datetime.now(timezone.utc).isoformat())

        except Exception as e:
            logger.error(f"Failed to run template {template_id}: {e}")
            self._record_failure(template_id, str(e))

    def _record_failure(self, template_id: int, error: str, filename: str = None, filepath: str = None):
        """Record a failed report generation."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failed_{template_id}_{timestamp}.pdf"
        if filepath is None:
            filepath = os.path.join(db.get_reports_directory(), filename)

        db.create_generated_report(
            template_id=template_id,
            filename=filename,
            file_path=filepath,
            status="failed",
            error_message=error[:500]  # Truncate long errors
        )
        logger.error(f"Report generation failed for template {template_id}: {error}")


def get_scheduler() -> Optional[ReportScheduler]:
    """Get the global scheduler instance."""
    return _scheduler_instance


def start_scheduler(api_token: str, check_interval_minutes: int = 60) -> ReportScheduler:
    """Start the global scheduler instance."""
    global _scheduler_instance

    with _scheduler_lock:
        if _scheduler_instance is not None and _scheduler_instance.is_running():
            logger.info("Scheduler already running, updating API token")
            _scheduler_instance.api_token = api_token
            return _scheduler_instance

        _scheduler_instance = ReportScheduler(
            check_interval_minutes=check_interval_minutes,
            api_token=api_token
        )
        _scheduler_instance.start()
        return _scheduler_instance


def stop_scheduler():
    """Stop the global scheduler instance."""
    global _scheduler_instance

    with _scheduler_lock:
        if _scheduler_instance is not None:
            _scheduler_instance.stop()
            _scheduler_instance = None


def run_template_now(template_id: int, api_token: str) -> tuple[bool, Optional[str]]:
    """
    Run a specific template immediately (ad-hoc execution).

    Args:
        template_id: ID of the template to run
        api_token: API token for Sysdig API

    Returns:
        Tuple of (success, error_message)
    """
    template = db.get_template(template_id)
    if not template:
        return False, "Template not found"

    try:
        config = yaml.safe_load(template['config_yaml'])
        region = config.get('global', {}).get('region', 'au1')
        orientation = config.get('global', {}).get('orientation', 'portrait')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = template['name'].replace(' ', '_').replace('/', '_')
        filename = f"{safe_name}_{timestamp}.pdf"
        filepath = os.path.join(db.get_reports_directory(), filename)

        os.makedirs(db.get_reports_directory(), exist_ok=True)

        success, error = generate_report(
            config=config,
            output_path=filepath,
            api_token=api_token,
            region=region,
            orientation=orientation
        )

        if success:
            file_size = os.path.getsize(filepath)
            db.create_generated_report(
                template_id=template_id,
                filename=filename,
                file_path=filepath,
                status="success",
                file_size=file_size
            )
            return True, None
        else:
            db.create_generated_report(
                template_id=template_id,
                filename=filename,
                file_path=filepath,
                status="failed",
                error_message=error
            )
            return False, error

    except Exception as e:
        return False, str(e)
