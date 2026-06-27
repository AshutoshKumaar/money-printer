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
                
        except Exception as exc:
            if self.logger:
                self.logger.warning("Telemetry file saving failed: %s", exc)


# Global instance
telemetry_tracker = TelemetryTracker()
