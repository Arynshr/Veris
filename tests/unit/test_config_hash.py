import shutil

from veris.core.config import config_hash


def test_hash_is_stable_and_ignores_secrets(settings_factory, prod_settings):
    a = settings_factory(groq_api_key="one")
    b = settings_factory(groq_api_key="two")
    assert config_hash(a) == config_hash(b) == config_hash(a)


def test_hash_changes_with_settings_and_policy_files(settings_factory, tmp_path):
    base = config_hash(settings_factory())
    assert config_hash(settings_factory(lookback_years=7)) != base
    tax = tmp_path / "taxonomy.yaml"
    shutil.copyfile(settings_factory().taxonomy_path, tax)
    tax.write_text(tax.read_text() + "\n# edited\n")
    assert config_hash(settings_factory(taxonomy_path=tax)) != base
    safety = tmp_path / "safety.yaml"
    shutil.copyfile(settings_factory().safety_path, safety)
    safety.write_text(safety.read_text() + "\n# edited\n")
    assert config_hash(settings_factory(safety_path=safety)) != base
