from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ApiRequestRecord:
    timestamp: str
    stage: str  # research, verification, story, scene, visual, voice, upload
    provider: str
    model: str
    endpoint: str
    request_id: str | None
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    images_requested: int
    images_returned: int
    attempt_number: int
    retry_count: int
    status_code: int
    latency: float
    cache_hit: bool
    response_size_bytes: int
    scene_index: int | None = None


class TelemetryTracker:
    """Tracks and aggregates external API usage across all pipeline stages safely."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger
        self.records: list[ApiRequestRecord] = []
        self.stage_timings: dict[str, dict] = {}
        self.memory_points: dict[str, float] = {}
        self.peak_memory: float = 0.0
        self.retries: dict[str, dict] = {
            "Gemini": {"total_retries": 0, "reasons": [], "recovered": True, "fallback": False},
            "Scene Planner": {"total_retries": 0, "reasons": [], "recovered": True, "fallback": False},
            "Visual": {"total_retries": 0, "reasons": [], "recovered": True, "fallback": False},
            "Upload": {"total_retries": 0, "reasons": [], "recovered": True, "fallback": False},
        }
        self.fallbacks: dict[str, int] = {
            "fallback_scene_package": 0,
            "cached_image": 0,
            "placeholder_image": 0,
            "offline_topic": 0,
            "upload_retry": 0,
            "offline_script": 0,
        }

    def record_stage_timing(self, stage: str, start_time: float, end_time: float) -> None:
        try:
            from datetime import datetime, timezone
            elapsed = (end_time - start_time) * 1000.0
            self.stage_timings[stage] = {
                "start_time": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                "end_time": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
                "elapsed_ms": round(elapsed, 2),
            }
        except Exception:
            pass

    def record_memory(self, point_name: str) -> None:
        try:
            import psutil
            mem = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
            self.memory_points[point_name] = mem
            if mem > self.peak_memory:
                self.peak_memory = mem
        except Exception:
            pass

    def record_retry(
        self,
        module: str,
        reason: str,
        recovered: bool = True,
        fallback: bool = False,
    ) -> None:
        try:
            if module not in self.retries:
                self.retries[module] = {"total_retries": 0, "reasons": [], "recovered": True, "fallback": False}
            self.retries[module]["total_retries"] += 1
            if reason and reason not in self.retries[module]["reasons"]:
                self.retries[module]["reasons"].append(reason)
            self.retries[module]["recovered"] = recovered
            self.retries[module]["fallback"] = fallback
        except Exception:
            pass

    def record_fallback(self, fallback_type: str) -> None:
        try:
            self.fallbacks[fallback_type] = self.fallbacks.get(fallback_type, 0) + 1
        except Exception:
            pass

    def record(
        self,
        stage: str,
        provider: str,
        model: str,
        endpoint: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        images_requested: int = 0,
        images_returned: int = 0,
        attempt_number: int = 1,
        retry_count: int = 0,
        status_code: int = 200,
        latency: float = 0.0,
        cache_hit: bool = False,
        response_size_bytes: int = 0,
        request_id: str | None = None,
        provider_request_id: str | None = None,
        scene_index: int | None = None,
    ) -> None:
        try:
            record = ApiRequestRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                stage=stage,
                provider=provider,
                model=model,
                endpoint=endpoint,
                request_id=request_id,
                provider_request_id=provider_request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                images_requested=images_requested,
                images_returned=images_returned,
                attempt_number=attempt_number,
                retry_count=retry_count,
                status_code=status_code,
                latency=round(latency, 3),
                cache_hit=cache_hit,
                response_size_bytes=response_size_bytes,
                scene_index=scene_index,
            )
            self.records.append(record)
        except Exception as exc:
            if self.logger:
                self.logger.warning("Telemetry recording failed silently: %s", exc)

    def generate_summary(
        self,
        run_id: str,
        scene_plan: ScenePlanManifest | None = None,
        visual_assets: VisualAssetManifest | None = None,
    ) -> dict:
        try:
            total_requests = len(self.records)
            gemini_requests = sum(1 for r in self.records if r.provider.lower() == "google" and "imagen" not in r.model.lower())
            imagen_requests = sum(1 for r in self.records if r.provider.lower() == "google" and "imagen" in r.model.lower())
            pexels_requests = sum(1 for r in self.records if r.provider.lower() == "pexels")
            pixabay_requests = sum(1 for r in self.records if r.provider.lower() == "pixabay")
            edge_tts_requests = sum(1 for r in self.records if r.provider.lower() == "microsoft" or "tts" in r.endpoint.lower())
            youtube_requests = sum(1 for r in self.records if r.provider.lower() == "youtube" or "youtube" in r.provider.lower())
            
            cache_hits = sum(1 for r in self.records if r.cache_hit)
            cache_misses = total_requests - cache_hits
            
            total_input_tokens = sum(r.input_tokens for r in self.records)
            total_output_tokens = sum(r.output_tokens for r in self.records)
            
            total_latency = sum(r.latency for r in self.records)
            average_latency_ms = (total_latency / total_requests * 1000.0) if total_requests > 0 else 0.0

            ai_images_used = 0
            stock_images_used = 0
            cache_images_used = 0
            ai_percentage = 0.0
            scene_priority_distribution = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            
            if visual_assets and visual_assets.assets:
                for asset in visual_assets.assets:
                    provider_lower = (asset.provider or "").lower()
                    if provider_lower == "aiimage":
                        ai_images_used += 1
                    elif provider_lower in ["pexels", "pixabay"]:
                        stock_images_used += 1
                    elif provider_lower == "cache":
                        cache_images_used += 1
                
                total_assets = len(visual_assets.assets)
                if total_assets > 0:
                    ai_percentage = round((ai_images_used / total_assets) * 100.0, 2)
            
            if scene_plan and scene_plan.scenes:
                for scene in scene_plan.scenes:
                    p = getattr(scene, "priority", "MEDIUM").upper()
                    scene_priority_distribution[p] = scene_priority_distribution.get(p, 0) + 1
            
            api_stats = {}
            providers = {
                "Gemini": lambda r: r.provider.lower() in ("google", "gemini") and "imagen" not in r.model.lower(),
                "Pexels": lambda r: r.provider.lower() == "pexels",
                "Pixabay": lambda r: r.provider.lower() == "pixabay",
                "Edge TTS": lambda r: r.provider.lower() in ("microsoft", "edge_tts") or "tts" in r.endpoint.lower(),
                "YouTube": lambda r: r.provider.lower() in ("youtube", "google_youtube"),
            }
            for name, filter_fn in providers.items():
                p_records = [r for r in self.records if filter_fn(r)]
                reqs = len(p_records)
                successes = sum(1 for r in p_records if r.status_code in (200, 201))
                failures = reqs - successes
                retries_sum = sum(r.retry_count for r in p_records)
                avg_lat = (sum(r.latency for r in p_records) / reqs) if reqs > 0 else 0.0
                hit_rate = (sum(1 for r in p_records if r.cache_hit) / reqs * 100.0) if reqs > 0 else 0.0
                api_stats[name] = {
                    "requests": reqs,
                    "success": successes,
                    "failure": failures,
                    "retries": retries_sum,
                    "average_latency_seconds": round(avg_lat, 3),
                    "cache_hit_rate_pct": round(hit_rate, 2),
                }

            return {
                "run_id": run_id,
                "total_requests": total_requests,
                "gemini_requests": gemini_requests,
                "imagen_requests": imagen_requests,
                "pexels_requests": pexels_requests,
                "pixabay_requests": pixabay_requests,
                "edge_tts_requests": edge_tts_requests,
                "youtube_requests": youtube_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "average_latency_ms": round(average_latency_ms, 2),
                "ai_images_used": ai_images_used,
                "stock_images_used": stock_images_used,
                "cache_images_used": cache_images_used,
                "ai_percentage": ai_percentage,
                "scene_priority_distribution": scene_priority_distribution,
                "timings": self.stage_timings,
                "memory": self.memory_points,
                "peak_memory_mb": self.peak_memory,
                "retries": self.retries,
                "fallbacks": self.fallbacks,
                "api_statistics": api_stats,
            }
        except Exception as exc:
            if self.logger:
                self.logger.warning("Telemetry summary generation failed: %s", exc)
            return {"run_id": run_id, "error": str(exc)}

    def save_all(
        self,
        debug_dir: Path,
        run_id: str,
        scene_plan: ScenePlanManifest | None = None,
        visual_assets: VisualAssetManifest | None = None,
    ) -> None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. api_usage.json
            usage_payload = {
                "schema_version": "1.0.0",
                "pipeline_version": "1.0.0",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "records": [asdict(r) for r in self.records]
            }
            with open(debug_dir / "api_usage.json", "w", encoding="utf-8") as f:
                json.dump(usage_payload, f, indent=2, ensure_ascii=False)
                
            # 2. api_summary.json
            summary_payload = self.generate_summary(run_id, scene_plan, visual_assets)
            # Add versioning info
            summary_payload = {
                "schema_version": "1.0.0",
                "pipeline_version": "1.0.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **summary_payload
            }
            with open(debug_dir / "api_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2, ensure_ascii=False)
                
            # 3. request_timeline.json
            # Sorted by timestamp
            sorted_records = sorted(self.records, key=lambda r: r.timestamp)
            timeline_payload = {
                "schema_version": "1.0.0",
                "pipeline_version": "1.0.0",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "timeline": [asdict(r) for r in sorted_records]
            }
            with open(debug_dir / "request_timeline.json", "w", encoding="utf-8") as f:
                json.dump(timeline_payload, f, indent=2, ensure_ascii=False)

            # 4. performance.json
            perf_payload = {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stages": self.stage_timings,
            }
            with open(debug_dir / "performance.json", "w", encoding="utf-8") as f:
                json.dump(perf_payload, f, indent=2, ensure_ascii=False)

            # 5. pipeline_summary.json
            pipeline_summary_data = {
                "schema_version": "1.0.0",
                "pipeline_version": "1.0.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **summary_payload
            }
            with open(debug_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
                json.dump(pipeline_summary_data, f, indent=2, ensure_ascii=False)
                
        except Exception as exc:
            if self.logger:
                self.logger.warning("Telemetry file saving failed: %s", exc)

    def print_final_summary(
        self,
        run_id: str,
        topic: str,
        category: str,
        duration: float,
        video_length: float,
        upload_success: bool,
        debug_folder: str,
        logger: logging.Logger,
    ) -> None:
        try:
            total_api_calls = len(self.records)
            total_retries = sum(v["total_retries"] for v in self.retries.values())
            total_fallbacks = sum(self.fallbacks.values())
            
            summary_text = (
                "\nPipeline Summary\n"
                "----------------\n"
                f"Run ID: {run_id}\n"
                f"Topic: {topic}\n"
                f"Category: {category}\n"
                f"Duration: {duration:.2f}s\n"
                f"Total API Calls: {total_api_calls}\n"
                f"Retries: {total_retries}\n"
                f"Fallbacks: {total_fallbacks}\n"
                f"Peak Memory: {self.peak_memory:.2f} MB\n"
                f"Video Length: {video_length:.2f}s\n"
                f"Upload Success: {'Yes' if upload_success else 'No'}\n"
                f"Debug Folder: {debug_folder}"
            )
            logger.info(summary_text)
        except Exception as e:
            if logger:
                logger.warning("Failed to print final summary: %s", e)


# Global instance
telemetry_tracker = TelemetryTracker()
