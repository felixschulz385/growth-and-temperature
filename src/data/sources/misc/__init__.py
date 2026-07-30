"""The misc split: osm, gadm, country_classifications.

docs/design/09-integrated-pipeline.md §7. Each is a standalone module
(imported directly by the registry, not re-exported here) so that importing
one doesn't pull in the others' dependencies.
"""
