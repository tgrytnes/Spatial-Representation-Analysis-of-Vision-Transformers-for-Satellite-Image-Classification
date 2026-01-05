from eurosat_vit_analysis import package_info


def test_package_info_contains_name() -> None:
    assert "Spatial Representation" in package_info()
