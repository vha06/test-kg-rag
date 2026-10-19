import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import (
    DomainConfig,
    ProcessingProfile,
    DomainRegistry,
    ProfileRegistry,
    ProfileResolutionError,
)


def test_01_domain_registration():
    d_reg = DomainRegistry()
    dom = DomainConfig(
        domain_id="dom_custom",
        name="Custom Domain",
        graph_namespace="ns_custom",
        default_profile_id="prof_custom_default",
    )
    d_reg.register_domain(dom)

    retrieved = d_reg.get_domain("dom_custom")
    assert retrieved is not None
    assert retrieved.name == "Custom Domain"
    assert retrieved.graph_namespace == "ns_custom"
    assert len(d_reg.list_domains()) == 1


def test_02_profile_resolution_exact_and_fallback():
    d_reg = DomainRegistry()
    d_reg.register_domain(
        DomainConfig(
            domain_id="finance",
            name="Finance Domain",
            graph_namespace="ns_fin",
            default_profile_id="prof_fin_default",
        )
    )

    p_reg = ProfileRegistry(domain_registry=d_reg)

    prof_invoice = ProcessingProfile(
        profile_id="prof_fin_invoice",
        domain_id="finance",
        document_type="invoice",
        chunking_strategy="sliding_window",
    )
    prof_default = ProcessingProfile(
        profile_id="prof_fin_default",
        domain_id="finance",
        document_type="general",
        chunking_strategy="block_preserving",
    )
    prof_global = ProcessingProfile(
        profile_id="prof_global",
        domain_id="*",
        document_type="*",
        chunking_strategy="fixed",
    )

    p_reg.register_profile(prof_invoice)
    p_reg.register_profile(prof_default)
    p_reg.register_profile(prof_global)

    # 1. Exact match by document_type
    resolved_1 = p_reg.resolve_profile(domain_id="finance", document_type="invoice")
    assert resolved_1.profile_id == "prof_fin_invoice"

    # 2. Match by direct profile_id
    resolved_2 = p_reg.resolve_profile(domain_id="finance", profile_id="prof_fin_invoice")
    assert resolved_2.profile_id == "prof_fin_invoice"

    # 3. Fallback to domain default when document_type has no specific profile
    resolved_3 = p_reg.resolve_profile(domain_id="finance", document_type="unregistered_contract")
    assert resolved_3.profile_id == "prof_fin_default"

    # 4. Fallback to global default when domain has no default profile
    resolved_4 = p_reg.resolve_profile(domain_id="unknown_domain", document_type="unknown_doc")
    assert resolved_4.profile_id == "prof_global"


def test_03_profile_resolution_disabled_and_error():
    p_reg = ProfileRegistry()
    prof_disabled = ProcessingProfile(
        profile_id="prof_disabled",
        domain_id="dom_test",
        document_type="doc_type_test",
        enabled=False,
        fallback_profile_id=None,
    )
    p_reg.register_profile(prof_disabled)

    with pytest.raises(ProfileResolutionError, match="Could not resolve an active processing profile"):
        p_reg.resolve_profile(domain_id="dom_test", document_type="doc_type_test")


def test_04_load_from_yaml():
    yaml_path = "config/platform_profiles.yaml"
    assert os.path.exists(yaml_path)

    d_reg = DomainRegistry()
    d_reg.load_from_yaml(yaml_path)

    p_reg = ProfileRegistry(domain_registry=d_reg)
    p_reg.load_from_yaml(yaml_path)

    # Check domains
    domains = d_reg.list_domains()
    assert len(domains) >= 2
    domain_ids = [d.domain_id for d in domains]
    assert "synth_domain_ops" in domain_ids
    assert "synth_domain_rd" in domain_ids

    # Check profile resolution for ops
    prof_ops = p_reg.resolve_profile("synth_domain_ops", document_type="operational_manual")
    assert prof_ops.profile_id == "profile_ops_standard"
    assert prof_ops.chunking_parameters["max_tokens"] == 256

    prof_inc = p_reg.resolve_profile("synth_domain_ops", document_type="incident_report")
    assert prof_inc.profile_id == "profile_ops_incident"
    assert prof_inc.chunking_strategy == "block_preserving"

    # Check profile resolution for rd
    prof_rd = p_reg.resolve_profile("synth_domain_rd", document_type="research_memo")
    assert prof_rd.profile_id == "profile_rd_standard"
    assert prof_rd.chunking_parameters["max_tokens"] == 512

    # Check list_profiles_for_domain
    ops_profiles = p_reg.list_profiles_for_domain("synth_domain_ops")
    ops_ids = [p.profile_id for p in ops_profiles]
    assert "profile_ops_standard" in ops_ids
    assert "profile_ops_incident" in ops_ids

