from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import psutil

from config import Settings
from core.models import GeneratedVideo
from core.pipeline import ShortsPipeline, ModularGeneratedVideo

# Modular engine imports
from research.research_engine import ResearchEngine
from research.models import ResearchPackageAdapter
from verification.verification_engine import VerificationEngine
from verification.models import VerificationAdapter
from story.story_engine import StoryEngine
from story.models import NarrativeAdapter
from scene.scene_engine import ScenePlanner
from scene.models import ScenePackageAdapter
from visual.visual_engine import VisualEngine
from visual.models import VisualPackageAdapter
from adapters.legacy_pipeline_adapter import LegacyPipelineAdapter

# Topic Intelligence & Analytics imports
from analytics.analytics_engine import AnalyticsEngine
from analytics.feedback_engine import FeedbackEngine
from topic.topic_history import TopicHistory
from topic.category_manager import CategoryManager
from topic.topic_engine import TopicEngine



class Director:
    """The single orchestration layer coordinating both modular and legacy pipelines."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.pipeline = ShortsPipeline(settings, logger)

    def execute_run(
        self,
        topic: str | None,
        *,
        dry_run: bool = False,
        generate_only: bool = False,
        use_existing_assets: bool = False,
    ) -> GeneratedVideo:
        self.logger.info("Director checking pipeline configuration flag...")
        
        # Feature Flag: Rollback to legacy if USE_MODULAR_PIPELINE is false
        if not self.settings.use_modular_pipeline:
            self.logger.warning("USE_MODULAR_PIPELINE is set to false. Falling back to legacy pipeline.")
            return self.pipeline.run(
                topic,
                dry_run=dry_run,
                generate_only=generate_only,
                use_existing_assets=use_existing_assets,
            )

        self.logger.info("Director executing modular run flow...")
        
        # Topic selection with Topic Intelligence
        analytics_engine = AnalyticsEngine(self.settings, self.logger)
        topic_history = TopicHistory(self.settings, self.logger, analytics_engine)
        category_manager = CategoryManager(self.settings, self.logger)
        topic_engine = TopicEngine(self.settings, self.logger, analytics_engine, topic_history, category_manager)
        
        topic_decision = topic_engine.decide_topic(topic)
        topic = topic_decision.topic
        self.logger.info("Resolved Topic: %s (Category: %s)", topic, topic_decision.category)
        
        paths = self.pipeline.storage.create_run(topic)
        
        # Reset telemetry records
        try:
            from core.telemetry import telemetry_tracker
            telemetry_tracker.records = []
        except Exception as e:
            self.logger.warning("Failed to reset telemetry records: %s", e)
        
        timings = {}
        partial_outputs = {}
        failed_stage = None
        error_msg = None
        tb_str = None
        
        try:
            # 1. Research Stage
            self.logger.info("Running stage: Research")
            t_start = time.time()
            research_engine = ResearchEngine(self.settings, self.logger)
            research_context = research_engine.research_topic(topic)
            t_diff = round(time.time() - t_start, 2)
            timings["research_time"] = t_diff
            partial_outputs["research"] = research_context.to_dict()
            self._save_stage_json("research.json", research_context.to_dict(), paths.run_id)

            # 2. Verification Stage
            self.logger.info("Running stage: Verification")
            t_start = time.time()
            verification_engine = VerificationEngine(self.settings, self.logger)
            verification_report = verification_engine.verify(ResearchPackageAdapter(research_context))
            t_diff = round(time.time() - t_start, 2)
            timings["verification_time"] = t_diff
            partial_outputs["verification"] = verification_report.to_dict()
            self._save_stage_json("verification.json", verification_report.to_dict(), paths.run_id)

            # 3. Story Stage
            self.logger.info("Running stage: Story")
            t_start = time.time()
            story_engine = StoryEngine(self.settings, self.logger)
            narrative_script = story_engine.write_story(VerificationAdapter(verification_report))
            t_diff = round(time.time() - t_start, 2)
            timings["story_time"] = t_diff
            partial_outputs["story"] = narrative_script.to_dict()
            self._save_stage_json("story.json", narrative_script.to_dict(), paths.run_id)

            # 4. Scene Planning Stage
            self.logger.info("Running stage: Scene Planning")
            t_start = time.time()
            scene_planner = ScenePlanner(self.settings, self.logger)
            scene_plan = scene_planner.plan(
                NarrativeAdapter(narrative_script),
                research=research_context,
                verified=verification_report,
            )
            t_diff = round(time.time() - t_start, 2)
            timings["scene_planning_time"] = t_diff
            partial_outputs["scene"] = scene_plan.to_dict()
            self._save_stage_json("scene.json", scene_plan.to_dict(), paths.run_id)
            scene_plan_adapter = ScenePackageAdapter(scene_plan)

            # 5. Visual Asset Resolution Stage (Only runs if not dry_run)
            visual_assets = None
            visual_package = None
            if not dry_run:
                self.logger.info("Running stage: Visual")
                t_start = time.time()
                visual_engine = VisualEngine(self.settings, self.logger)
                visual_package = visual_engine.resolve_assets(scene_plan_adapter)
                visual_assets = VisualPackageAdapter(visual_package)
                t_diff = round(time.time() - t_start, 2)
                timings["visual_time"] = t_diff
                
                # Traceable visual asset records
                trace_assets = []
                for scene in scene_plan_adapter.scenes:
                    asset = next((a for a in visual_assets.assets if a.scene_index == scene.scene_index), None)
                    if asset:
                        final_prompt = (
                            f"{scene.ai_image_prompt}. Vertical 9:16 composition, cinematic, realistic, high contrast, "
                            "sharp subject, no watermark, no subtitles, no UI text, photorealistic, "
                            "professional cinematography, movie quality."
                        ) if asset.provider == "aiimage" else (scene.search_query or scene.visual_description)
                        
                        trace_assets.append({
                            "provider": asset.provider,
                            "asset_type": asset.asset_type,
                            "original_prompt": scene.ai_image_prompt if asset.provider == "aiimage" else scene.search_query,
                            "final_prompt": final_prompt,
                            "generation_time": asset.generation_time,
                            "quality_score": asset.quality_score,
                            "confidence": asset.confidence,
                            "cache_hit": asset.cache_hit,
                            "file_path": asset.file_path,
                        })
                
                visual_payload = visual_package.to_dict()
                partial_outputs["visual"] = visual_payload
                self._save_stage_json("visual.json", visual_payload, paths.run_id)
            else:
                timings["visual_time"] = 0.0
                partial_outputs["visual"] = {}

            # 6. Legacy Adapter Layer
            self.logger.info("Running stage: Legacy Adapter Mapping")
            script = LegacyPipelineAdapter.adapt_script(topic, NarrativeAdapter(narrative_script), scene_plan_adapter, visual_assets)
            image_paths = LegacyPipelineAdapter.extract_image_paths(scene_plan_adapter, visual_assets) if visual_assets else []

            # 7. Execute Voiceovers, Rendering, and Uploading via ShortsPipeline
            result = self.pipeline.run_modular(
                script,
                image_paths,
                paths,
                dry_run=dry_run,
                generate_only=generate_only,
                use_existing_assets=use_existing_assets,
                narrative_package=narrative_script,
                scene_package=scene_plan,
                visual_package=visual_package,
            )

            # Extract pipeline execution times from run_modular timings dict
            pipeline_timings = getattr(result, "timings", {})
            timings["voice_time"] = pipeline_timings.get("voice_time", 0.0)
            timings["render_time"] = pipeline_timings.get("render_time", 0.0)
            timings["upload_time"] = pipeline_timings.get("upload_time", 0.0)

            # Calculate total time
            total_time = round(sum(timings.values()), 2)
            timings["total_time"] = total_time

            # Gather performance metrics
            process_mem = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
            cache_hits = sum(1 for asset in visual_assets.assets if asset.cache_hit) if visual_assets else 0
            cache_misses = sum(1 for asset in visual_assets.assets if not asset.cache_hit) if visual_assets else 0

            metrics = {
                "memory_usage_mb": process_mem,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
            }

            render_payload = {
                "video_path": str(result.video_path) if result.video_path else None,
                "metadata_path": str(result.metadata_path),
                "youtube_url": result.youtube_url,
                "duration": float(getattr(self.pipeline.video, "last_rendered_duration", 0.0) or 0.0), # fallback metadata
            }
            self._save_stage_json("render.json", render_payload, paths.run_id)

            # 8. Analytics Save Stage
            self.logger.info("Running stage: Analytics Save")
            t_start_save = time.time()
            feedback_engine = FeedbackEngine(self.settings, self.logger)
            feedback_engine.save_run_performance(result, topic_decision)
            timings["analytics_save_time"] = round(time.time() - t_start_save, 2)

            # Recalculate total time including analytics save stage
            timings.pop("total_time", None)
            total_time = round(sum(timings.values()), 2)
            timings["total_time"] = total_time

            summary_payload = {
                "topic": topic,
                "status": "success",
                "timings": timings,
                "metrics": metrics,
            }
            self._save_stage_json("summary.json", summary_payload, paths.run_id)

            return result

        except Exception as exc:
            failed_stage = failed_stage or "unknown"
            if "research_time" not in timings:
                failed_stage = "Research"
            elif "verification_time" not in timings:
                failed_stage = "Verification"
            elif "story_time" not in timings:
                failed_stage = "Story"
            elif "scene_planning_time" not in timings:
                failed_stage = "Scene Planning"
            elif "visual_time" not in timings and not dry_run:
                failed_stage = "Visual"
            elif "voice_time" not in timings:
                failed_stage = "Voice"
            elif "render_time" not in timings:
                failed_stage = "Render"
            else:
                failed_stage = "Upload"

            error_msg = str(exc)
            tb_str = traceback.format_exc()

            self.logger.error("Stage %s failed with error: %s", failed_stage, error_msg)

            # Save failure.json if debug persist is enabled
            failure_payload = {
                "failed_stage": failed_stage,
                "error_message": error_msg,
                "stack_trace": tb_str,
                "timings": timings,
                "partial_outputs": partial_outputs,
            }
            self._save_stage_json("failure.json", failure_payload, paths.run_id)

            raise exc
        finally:
            try:
                from core.telemetry import telemetry_tracker
                debug_dir = self.settings.storage_dir / "debug" / paths.run_id
                
                # Fetch local variables if defined
                local_scene_plan = locals().get("scene_plan", None)
                local_visual_assets = locals().get("visual_assets", None)
                
                telemetry_tracker.save_all(debug_dir, paths.run_id, scene_plan=local_scene_plan, visual_assets=local_visual_assets)
                
                # Print and log telemetry summary
                summary = telemetry_tracker.generate_summary(paths.run_id, scene_plan=local_scene_plan, visual_assets=local_visual_assets)
                total_retries = sum(r.retry_count for r in telemetry_tracker.records)
                total_images_requested = sum(r.images_requested for r in telemetry_tracker.records)
                total_images_returned = sum(r.images_returned for r in telemetry_tracker.records)
                avg_images = (total_images_returned / summary["imagen_requests"]) if summary["imagen_requests"] > 0 else 0.0
                
                report = (
                    "\n=========================================\n"
                    "          API TELEMETRY REPORT          \n"
                    "=========================================\n"
                    f"Total Requests:      {summary.get('total_requests', 0)}\n"
                    f"Cache Hits:          {summary.get('cache_hits', 0)}\n"
                    f"Cache Misses:        {summary.get('cache_misses', 0)}\n"
                    "-----------------------------------------\n"
                    f"Gemini Requests:     {summary.get('gemini_requests', 0)}\n"
                    f"Imagen Requests:     {summary.get('imagen_requests', 0)}\n"
                    f"Pexels Requests:     {summary.get('pexels_requests', 0)}\n"
                    f"Pixabay Requests:    {summary.get('pixabay_requests', 0)}\n"
                    f"Edge TTS Requests:   {summary.get('edge_tts_requests', 0)}\n"
                    f"YouTube Requests:    {summary.get('youtube_requests', 0)}\n"
                    "-----------------------------------------\n"
                    f"Total Input Tokens:  {summary.get('total_input_tokens', 0)}\n"
                    f"Total Output Tokens: {summary.get('total_output_tokens', 0)}\n"
                    f"Images Requested:    {total_images_requested}\n"
                    f"Images Returned:     {total_images_returned}\n"
                    f"Average Images/Req:  {avg_images:.2f}\n"
                    f"Total Retries:       {total_retries}\n"
                    f"Average Latency:     {summary.get('average_latency_ms', 0.0):.2f} ms\n"
                    "-----------------------------------------\n"
                    "        COST OPTIMIZATION METRICS       \n"
                    "-----------------------------------------\n"
                    f"AI Images Used:      {summary.get('ai_images_used', 0)}\n"
                    f"Stock Images Used:   {summary.get('stock_images_used', 0)}\n"
                    f"Cache Images Used:   {summary.get('cache_images_used', 0)}\n"
                    f"AI Image Percentage: {summary.get('ai_percentage', 0.0):.2f}%\n"
                    f"Priority Distribution: {summary.get('scene_priority_distribution', {})}\n"
                    "-----------------------------------------\n"
                    "             OUTPUT LOCATIONS            \n"
                    "-----------------------------------------\n"
                    f"Debug Folder:        {debug_dir}\n"
                    f"API Usage Log:       {debug_dir / 'api_usage.json'}\n"
                    f"API Summary Log:     {debug_dir / 'api_summary.json'}\n"
                    "=========================================\n"
                )
                print(report)
                self.logger.info(report)
            except Exception as e:
                self.logger.warning("Telemetry save or reporting failed: %s", e)

    def execute_generate_only(
        self,
        topic: str | None,
        *,
        use_existing_assets: bool = False,
    ) -> GeneratedVideo:
        self.logger.info("Director executing generate-only flow...")
        return self.execute_run(
            topic,
            dry_run=False,
            generate_only=True,
            use_existing_assets=use_existing_assets,
        )

    def execute_upload_only(
        self,
        video_path: Path,
        metadata_path: Path,
        thumbnail_path: Path | None = None,
    ) -> str:
        self.logger.info("Director executing upload-only flow...")
        return self.pipeline.upload_existing(
            video_path,
            metadata_path,
            thumbnail_path=thumbnail_path,
        )

    def scheduled_job(self, video_type: str = "short") -> None:
        self.logger.info("Director executing scheduled job for %s video...", video_type)
        import json
        import zoneinfo
        from dataclasses import replace
        
        try:
            self.settings.validate(require_youtube=True)
            is_long = (video_type == "long")
            orig_settings = self.settings
            
            if is_long:
                long_settings = replace(
                    self.settings,
                    shorts_target_seconds=180,
                    shorts_max_seconds=240,
                    min_segments=15,
                    max_segments=20,
                    video_resolution=(1280, 720)
                )
                self.settings = long_settings
                self.pipeline.settings = long_settings
                self.pipeline.images.settings = long_settings
                self.pipeline.voice.settings = long_settings
                self.pipeline.video.settings = long_settings
                self.pipeline.youtube.settings = long_settings

            topic = None
            try:
                # Topic selection with Topic Intelligence
                analytics_engine = AnalyticsEngine(self.settings, self.logger)
                topic_history = TopicHistory(self.settings, self.logger, analytics_engine)
                category_manager = CategoryManager(self.settings, self.logger)
                topic_engine = TopicEngine(self.settings, self.logger, analytics_engine, topic_history, category_manager)
                
                topic_decision = topic_engine.decide_topic(topic=None)
                topic = topic_decision.topic
                
                # Check duplicate prevention
                history_file = self.settings.storage_dir / "upload_history.json"
                if history_file.exists():
                    try:
                        history_data = json.loads(history_file.read_text(encoding="utf-8"))
                        if any(item.get("topic") == topic and item.get("upload_status") == "success" for item in history_data):
                            self.logger.warning("Duplicate upload prevented for topic: %s", topic)
                            self._append_scheduler_log("duplicate_prevented", f"Topic: {topic}")
                            return
                    except Exception:
                        pass
                
                result = self.execute_run(
                    topic=topic,
                    dry_run=False,
                    generate_only=False,
                    use_existing_assets=False,
                )
                
                # Write to upload_history.json
                self._append_upload_history({
                    "video_id": result.youtube_url.split("v=")[-1] if result.youtube_url else "unknown",
                    "topic": topic,
                    "publish_time": self._get_ist_time_str(),
                    "upload_status": "success" if result.youtube_url else "failed",
                    "file_path": str(result.video_path) if result.video_path else ""
                })
                self._append_scheduler_log("job_completed", f"Type: {video_type}, Topic: {topic}")
            except Exception as e:
                self.logger.exception("Scheduled execution failed")
                self._append_failed_upload({
                    "topic": topic if topic else "unknown",
                    "publish_time": self._get_ist_time_str(),
                    "error": str(e),
                    "file_path": str(result.video_path) if ('result' in locals() and result.video_path) else ""
                })
                self._append_scheduler_log("job_failed", f"Type: {video_type}, Error: {str(e)}")
                raise
            finally:
                if is_long:
                    # Restore original settings
                    self.settings = orig_settings
                    self.pipeline.settings = orig_settings
                    self.pipeline.images.settings = orig_settings
                    self.pipeline.voice.settings = orig_settings
                    self.pipeline.video.settings = orig_settings
                    self.pipeline.youtube.settings = orig_settings
        except Exception as exc:
            self.pipeline.notifications.send("Hindi Shorts automation failed", str(exc), success=False)

    def _get_ist_time_str(self) -> str:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo("Asia/Kolkata")
            return datetime.now(tz).isoformat()
        except Exception:
            from datetime import timezone, timedelta
            tz = timezone(timedelta(hours=5, minutes=30))
            return datetime.now(tz).isoformat()

    def _append_upload_history(self, record: dict) -> None:
        import json
        history_file = self.settings.storage_dir / "upload_history.json"
        history = []
        if history_file.exists():
            try:
                history = json.loads(history_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        history.append(record)
        history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_scheduler_log(self, event: str, details: str = "") -> None:
        import json
        log_file = self.settings.storage_dir / "scheduler_log.json"
        logs = []
        if log_file.exists():
            try:
                logs = json.loads(log_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        logs.append({
            "timestamp": self._get_ist_time_str(),
            "event": event,
            "details": details
        })
        log_file.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_failed_upload(self, record: dict) -> None:
        import json
        failed_file = self.settings.storage_dir / "failed_uploads.json"
        failed = []
        if failed_file.exists():
            try:
                failed = json.loads(failed_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        failed.append(record)
        failed_file.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_stage_json(self, filename: str, data: dict, run_id: str) -> None:
        if self.settings.pipeline_debug_persist:
            try:
                debug_dir = self.settings.storage_dir / "debug" / run_id
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Combine metadata with stage data at the top level
                payload = {
                    "schema_version": "1.0.0",
                    "pipeline_version": "1.0.0",
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **data
                }
                
                self.pipeline.storage.save_json(debug_dir / filename, payload)
            except Exception as e:
                self.logger.warning("Failed to save debug JSON file %s: %s", filename, e)
