from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated
from visionlib.pipeline.settings import LogLevel, YamlConfigSettingsSource
from typing import Optional


class RedisConfig(BaseModel):
    host: str = 'localhost'
    port: Annotated[int, Field(ge=1, le=65536)] = 6379
    stream_id: str = 'stream1'
    stream_prefix: str = 'undefined'

class ModelConfig(BaseModel):
    start_training: bool = False
    time_interval_train_data: float = 24
    path_train_data: Path = Path("model/dataset")

    anomaly_threshold_override: Optional[float] = None
    parameters_path: Path = Path("model/parameters.json")
    weights_path: Path = Path("model/model_weights.pth")

class AnomalyDetectionConfig(BaseSettings):
    log_level: LogLevel = LogLevel.INFO
    redisIn: RedisConfig = RedisConfig()
    redisOut: RedisConfig = RedisConfig()
    prometheus_port: Annotated[int, Field(ge=1024, le=65536)] = 8000
    model: ModelConfig = ModelConfig()
    filtering: bool = True
    debug_plots: bool = False

    # `model_config` refers to the pydantic model and has nothing to do with `model` above
    model_config = SettingsConfigDict(env_nested_delimiter='__')
    
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls), file_secret_settings)
    

class ModelTrainingConfig(BaseModel):
    model_name: str = "MobileNet_v3_asymmetric_prob"
    model_architecture: str = "MobileNet_v3"
    output_distribution: str = "asymmetric"
    pixel_per_axis: int = 120
    time_diff_prediction: float = 1.0
    percentage_anomaly: float = 99.95
    framerate: float = 10.0