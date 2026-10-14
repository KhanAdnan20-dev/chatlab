"""Model Manager — central thread-safe lifecycle management for Gemma 4 and NLP models."""
import gc
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe Singleton registry for model access and lifecycle management."""

    def __init__(self) -> None:
        self._intent_classifier = None
        self._semantic_engine = None
        self._entity_extractor = None
        self._gemma_model = None
        self._gemma_processor = None
        self._deepseek_model = None
        self._deepseek_tokenizer = None
        self._active_model_name: str = "gemma"
        self._lock = threading.RLock()
        self._load_error: Optional[str] = None
        self._load_time_seconds: float = 0.0
        self._test_mode: bool = os.getenv("CHATLAB_TEST_MODE", "false").lower() == "true"


    def set_test_mode(self, enabled: bool) -> None:
        """Enables or disables test mode to avoid loading heavy LLM weights in unit tests."""
        self._test_mode = enabled

    @property
    def intent_classifier(self):
        if self._intent_classifier is None:
            from backend.app.services.intent_classifier import intent_classifier
            self._intent_classifier = intent_classifier
        return self._intent_classifier

    @property
    def semantic_engine(self):
        if self._semantic_engine is None:
            from backend.app.services.semantic_similarity import semantic_similarity_engine
            self._semantic_engine = semantic_similarity_engine
        return self._semantic_engine

    @property
    def entity_extractor(self):
        if self._entity_extractor is None:
            from backend.app.services.entity_extractor import entity_extractor
            self._entity_extractor = entity_extractor
        return self._entity_extractor

    # --- Gemma 4 E2B Instruct Properties & Lifecycle ---

    @property
    def gemma_available(self) -> bool:
        if self._test_mode:
            return True
        return self._gemma_model is not None and self._gemma_processor is not None

    @property
    def gemma_model(self):
        return self._gemma_model

    @property
    def gemma_processor(self):
        return self._gemma_processor

    def ensure_gemma_loaded(self) -> bool:
        """Thread-safe lazy loading of Google Gemma 4 E2B Instruct model and processor."""
        if self.gemma_available:
            return True

        with self._lock:
            # Double-checked locking
            if self.gemma_available:
                return True
            return self._load_gemma()

    def _load_gemma(self) -> bool:
        """Loads Gemma 4 E2B Instruct with 4-bit NF4 quantization on CUDA if available."""
        if self._test_mode:
            logger.info("Test mode active; bypassing Gemma model loading.")
            return True

        # Hot-swap safeguard: Evict DeepSeek from VRAM first to protect 6GB GPU limit
        self.unload_deepseek()

        try:
            import torch

            from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
            from backend.app.config import settings

            model_id = settings.GEMMA_MODEL_ID
            logger.info("Loading Gemma processor from '%s'...", model_id)
            t0 = time.time()

            self._gemma_processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=True,
            )

            bnb_config = None
            if settings.USE_4BIT_QUANTIZATION:
                if torch.cuda.is_available():
                    logger.info("Configuring 4-bit NF4 quantization with float16 compute dtype and CPU offload...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                else:
                    logger.warning("CUDA unavailable; 4-bit quantization skipped (CPU fallback).")

            # On single-GPU systems (e.g. RTX 4050 Laptop), mapping directly to {'': 0}
            # avoids accelerate's aggressive CPU/disk offloading heuristic while keeping
            # 4-bit NF4 weights squarely in GPU VRAM.
            if torch.cuda.is_available():
                device_target = {"": 0} if settings.DEVICE_MAP in ("auto", "cuda", "cuda:0") else settings.DEVICE_MAP
            else:
                device_target = None

            logger.info("Loading Gemma 4 E2B Instruct model from '%s' (device_map=%s, attn=sdpa)...", model_id, device_target)
            self._gemma_model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_target,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="sdpa",
                local_files_only=True,
                trust_remote_code=True,
            )
            self._gemma_model.eval()

            self._load_time_seconds = round(time.time() - t0, 2)
            self._active_model_name = "gemma"
            self._load_error = None
            device_name = str(getattr(self._gemma_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
            logger.info(
                "Gemma 4 model loaded successfully in %.2fs on device %s.",
                self._load_time_seconds,
                device_name,
            )
            return True

        except Exception as exc:
            import torch
            if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", tuple())):
                self._load_error = f"Insufficient GPU VRAM to load Gemma 4: {exc}"
            else:
                self._load_error = f"Failed to load Gemma 4 model: {exc}"

            logger.error(self._load_error, exc_info=True)
            self.unload_gemma()
            return False

    def unload_gemma(self) -> None:
        """Unloads Gemma model and processor from memory and frees GPU VRAM."""
        with self._lock:
            if self._gemma_model is not None or self._gemma_processor is not None:
                logger.info("Unloading Gemma model and processor from memory...")
                del self._gemma_model
                del self._gemma_processor
                self._gemma_model = None
                self._gemma_processor = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

    # --- DeepSeek-R1 Distill Qwen 1.5B Properties & Lifecycle ---

    @property
    def deepseek_available(self) -> bool:
        if self._test_mode:
            return True
        return self._deepseek_model is not None and self._deepseek_tokenizer is not None

    @property
    def deepseek_model(self):
        return self._deepseek_model

    @property
    def deepseek_tokenizer(self):
        return self._deepseek_tokenizer

    @property
    def active_model_name(self) -> str:
        return self._active_model_name

    def ensure_deepseek_loaded(self) -> bool:
        """Thread-safe lazy loading of DeepSeek-R1-Distill-Qwen-1.5B model and tokenizer."""
        if self.deepseek_available:
            return True

        with self._lock:
            if self.deepseek_available:
                return True
            return self._load_deepseek()

    def _load_deepseek(self) -> bool:
        """Loads DeepSeek-R1 1.5B with 4-bit NF4 quantization on CUDA if available."""
        if self._test_mode:
            logger.info("Test mode active; bypassing DeepSeek model loading.")
            return True

        # Hot-swap safeguard: Evict Gemma from VRAM first to protect 6GB GPU limit
        self.unload_gemma()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from backend.app.config import settings

            model_id = settings.DEEPSEEK_MODEL_ID
            logger.info("Loading DeepSeek tokenizer from '%s'...", model_id)
            t0 = time.time()

            self._deepseek_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=True,
            )

            bnb_config = None
            if settings.USE_4BIT_QUANTIZATION and torch.cuda.is_available():
                logger.info("Configuring 4-bit NF4 quantization for DeepSeek-R1 1.5B...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )

            device_target = {"": 0} if torch.cuda.is_available() else None

            logger.info("Loading DeepSeek-R1 1.5B model from '%s' (device_map=%s, attn=sdpa)...", model_id, device_target)
            self._deepseek_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_target,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="sdpa",
                local_files_only=True,
                trust_remote_code=True,
            )
            self._deepseek_model.eval()

            self._load_time_seconds = round(time.time() - t0, 2)
            self._active_model_name = "deepseek"
            self._load_error = None
            device_name = str(getattr(self._deepseek_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu"))
            logger.info(
                "DeepSeek-R1 1.5B model loaded successfully in %.2fs on device %s.",
                self._load_time_seconds,
                device_name,
            )
            return True

        except Exception as exc:
            import torch
            if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", tuple())):
                self._load_error = f"Insufficient GPU VRAM to load DeepSeek-R1: {exc}"
            else:
                self._load_error = f"Failed to load DeepSeek-R1 model: {exc}"

            logger.error(self._load_error, exc_info=True)
            self.unload_deepseek()
            return False

    def unload_deepseek(self) -> None:
        """Unloads DeepSeek model and tokenizer from memory and frees GPU VRAM."""
        with self._lock:
            if self._deepseek_model is not None or self._deepseek_tokenizer is not None:
                logger.info("Unloading DeepSeek model and tokenizer from memory...")
                del self._deepseek_model
                del self._deepseek_tokenizer
                self._deepseek_model = None
                self._deepseek_tokenizer = None
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

    def switch_model(self, model_name: str) -> bool:
        """Hot-swaps active LLM engine between 'gemma' and 'deepseek'."""
        clean = (model_name or "gemma").lower()
        if "deepseek" in clean or "r1" in clean or "reason" in clean:
            return self.ensure_deepseek_loaded()
        return self.ensure_gemma_loaded()



    # --- Status Reporting ---

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive model lifecycle readiness, VRAM, and device status."""
        vram_mb = 0.0
        vram_reserved_mb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
                vram_reserved_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        except Exception:
            pass

        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_map_info = None
        if self._gemma_model is not None:
            dev = getattr(self._gemma_model, "device", None)
            if dev is not None:
                device_str = str(dev)
            device_map_info = getattr(self._gemma_model, "hf_device_map", {"": device_str})

        return {
            "model_name": "Gemma 4 E2B Instruct" if self._active_model_name == "gemma" else "DeepSeek-R1 Distill Qwen 1.5B",
            "active_model": self._active_model_name,
            "gemma_loaded": self.gemma_available,
            "deepseek_loaded": self.deepseek_available,
            "intent_classifier_loaded": self.intent_classifier.is_loaded,
            "device": device_str,
            "device_map": device_map_info,
            "vram_allocated_mb": vram_mb,
            "vram_reserved_mb": vram_reserved_mb,
            "load_time_seconds": self._load_time_seconds,
            "load_error": self._load_error,
        }



model_manager = ModelManager()
