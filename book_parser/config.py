"""
Central configuration for the Book Parser system.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """HuggingFace model configuration."""
    vision_model: str = "Qwen/Qwen2-VL-7B-Instruct"
    text_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    device: str = "auto"
    torch_dtype: str = "float16"
    max_new_tokens: int = 4096


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    workers: int = 4
    batch_size: int = 10
    dpi: int = 300  # PDF to image resolution
    image_format: str = "png"
    output_dir: str = "output"
    pages_dir: str = "output/pages"
    results_dir: str = "output/results"


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "question_bank")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class TopicMapping:
    """Page range to topic mapping."""
    mappings: dict = field(default_factory=lambda: {
        # Example: (start_page, end_page) -> topic
        # Customize based on your book's table of contents
        (1, 2): "general",
        (3, 50): "number_system",
        (51, 100): "hcf_lcm",
        (101, 150): "fractions_decimals",
        (151, 200): "simplification",
        (201, 250): "square_roots_cube_roots",
        (251, 300): "average",
        (301, 350): "problems_on_numbers",
        (351, 400): "problems_on_ages",
        (401, 450): "percentage",
        (451, 500): "profit_and_loss",
        (501, 550): "ratio_and_proportion",
        (551, 600): "partnership",
        (601, 650): "time_and_work",
        (651, 700): "pipes_and_cisterns",
        (701, 750): "time_and_distance",
        (751, 800): "trains",
        (801, 850): "boats_and_streams",
        (851, 900): "simple_interest",
        (901, 926): "compound_interest",
    })

    def get_topic(self, page_number: int) -> str:
        for (start, end), topic in self.mappings.items():
            if start <= page_number <= end:
                return topic
        return "general"


@dataclass
class AppConfig:
    """Root application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    topics: TopicMapping = field(default_factory=TopicMapping)

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        Path(self.pipeline.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pipeline.pages_dir).mkdir(parents=True, exist_ok=True)
        Path(self.pipeline.results_dir).mkdir(parents=True, exist_ok=True)
