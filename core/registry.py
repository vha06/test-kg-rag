from __future__ import annotations

import os
import yaml
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ProfileResolutionError(Exception):
    """Raised when a processing profile cannot be resolved for a document."""
    pass


class DomainConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    domain_id: str
    name: str
    description: str = ""
    enabled: bool = True
    schema_version: str = "1.0.0"
    graph_namespace: str
    default_profile_id: Optional[str] = None
    is_test_fixture: bool = True
    allowed_entity_types: List[str] = Field(default_factory=list)
    allowed_predicates: List[str] = Field(default_factory=list)

    @field_validator("domain_id", "graph_namespace")
    @classmethod
    def check_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"'{info.field_name}' must be non-empty.")
        return v.strip()


class ProcessingProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_id: str
    domain_id: str
    document_type: str
    profile_version: str = "1.0.0"
    parser_strategy: str = "generic_paragraph"
    chunking_strategy: str = "sliding_window"
    chunking_parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"max_tokens": 256, "chunk_overlap": 32}
    )
    metadata_extractors: List[str] = Field(default_factory=list)
    entity_schema_ref: Optional[str] = None
    relationship_schema_ref: Optional[str] = None
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    fallback_profile_id: Optional[str] = None
    enabled: bool = True

    @field_validator("profile_id", "domain_id", "document_type")
    @classmethod
    def check_non_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"'{info.field_name}' must be non-empty.")
        return v.strip()


class DomainRegistry:
    def __init__(self):
        self._domains: Dict[str, DomainConfig] = {}

    def register_domain(self, domain: DomainConfig) -> None:
        self._domains[domain.domain_id] = domain

    def get_domain(self, domain_id: str) -> Optional[DomainConfig]:
        return self._domains.get(domain_id)

    def list_domains(self, enabled_only: bool = True) -> List[DomainConfig]:
        if enabled_only:
            return [d for d in self._domains.values() if d.enabled]
        return list(self._domains.values())

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        raw_domains = data.get("domains", [])
        for item in raw_domains:
            self.register_domain(DomainConfig.model_validate(item))

    def load_from_yaml(self, yaml_path: str) -> None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Domain config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.load_from_dict(data)


class ProfileRegistry:
    def __init__(self, domain_registry: Optional[DomainRegistry] = None):
        self._profiles: Dict[str, ProcessingProfile] = {}
        self.domain_registry = domain_registry or DomainRegistry()

    def register_profile(self, profile: ProcessingProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def get_profile(self, profile_id: str) -> Optional[ProcessingProfile]:
        return self._profiles.get(profile_id)

    def list_profiles(
        self, domain_id: Optional[str] = None, enabled_only: bool = True
    ) -> List[ProcessingProfile]:
        res = list(self._profiles.values())
        if domain_id is not None:
            res = [p for p in res if p.domain_id in (domain_id, "*")]
        if enabled_only:
            res = [p for p in res if p.enabled]
        return res

    def resolve_profile(
        self,
        domain_id: str,
        document_type: Optional[str] = None,
        profile_id: Optional[str] = None,
    ) -> ProcessingProfile:
        """
        Phân giải profile phù hợp nhất theo thứ tự ưu tiên:
        1. profile_id trực tiếp (nếu còn enabled, hoặc theo fallback_profile_id của nó)
        2. Cặp (domain_id, document_type)
        3. domain.default_profile_id
        4. Global fallback profile (domain_id == '*')
        """
        # 1. Profile ID requested directly
        if profile_id:
            prof = self._profiles.get(profile_id)
            if prof and prof.enabled:
                return prof
            if prof and not prof.enabled and prof.fallback_profile_id:
                fb = self._profiles.get(prof.fallback_profile_id)
                if fb and fb.enabled:
                    return fb

        # 2. Match domain_id + document_type
        if document_type:
            for p in self._profiles.values():
                if p.enabled and p.domain_id == domain_id and p.document_type == document_type:
                    return p

        # 3. Domain default profile
        dom = self.domain_registry.get_domain(domain_id)
        if dom and dom.default_profile_id:
            default_p = self._profiles.get(dom.default_profile_id)
            if default_p and default_p.enabled:
                return default_p

        # 4. Check global fallback profile
        for p in self._profiles.values():
            if p.enabled and p.domain_id == "*":
                return p

        raise ProfileResolutionError(
            f"Could not resolve an active processing profile for domain='{domain_id}', "
            f"document_type='{document_type}', profile_id='{profile_id}'."
        )

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        raw_profiles = data.get("profiles", [])
        for item in raw_profiles:
            self.register_profile(ProcessingProfile.model_validate(item))

    def load_from_yaml(self, yaml_path: str) -> None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Profile config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.load_from_dict(data)
